import os
import pprint
import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from method_tvr.config import TestOptions
from method_tvr.model import ReLoCLNet
from method_tvr.start_end_dataset import StartEndEvalDataset, prepare_batch_inputs, start_end_collate
from standalone_eval.eval import eval_retrieval
from utils.basic_utils import save_json
from utils.temporal_nms import temporal_non_maximum_suppression
from utils.tensor_utils import find_max_triples_from_upper_triangle_product

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def filter_vcmr_by_nms(all_video_predictions, nms_threshold=0.6, max_before_nms=1000, max_after_nms=100,
                       score_col_idx=3):
    """Apply temporal NMS per video id and return globally ranked predictions."""
    predictions_by_video = defaultdict(list)
    for pred in all_video_predictions[:max_before_nms]:
        predictions_by_video[pred[0]].append(pred[1:])

    merged = []
    for video_idx, grouped_preds in predictions_by_video.items():
        kept = temporal_non_maximum_suppression(grouped_preds, nms_threshold=nms_threshold)
        merged.extend([[video_idx] + pred for pred in kept])

    merged = sorted(merged, key=lambda x: x[score_col_idx], reverse=True)[:max_after_nms]
    return merged


def post_processing_vcmr_nms(vcmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """Run NMS for each VCMR query result."""
    processed = []
    for item in vcmr_res:
        copied_item = dict(item)
        copied_item["predictions"] = filter_vcmr_by_nms(
            item["predictions"],
            nms_threshold=nms_thd,
            max_before_nms=max_before_nms,
            max_after_nms=max_after_nms,
        )
        processed.append(copied_item)
    return processed


def post_processing_svmr_nms(svmr_res, nms_thd=0.6, max_before_nms=1000, max_after_nms=100):
    """Run NMS for each SVMR query result."""
    processed = []
    for item in svmr_res:
        raw_predictions = item["predictions"][:max_before_nms]
        copied_item = dict(item)
        if len(raw_predictions) == 0:
            copied_item["predictions"] = []
            processed.append(copied_item)
            continue


        video_idx = raw_predictions[0][0]
        temporal_predictions = [[pred[1], pred[2], pred[3]] for pred in raw_predictions]
        temporal_after_nms = temporal_non_maximum_suppression(
            temporal_predictions,
            nms_threshold=nms_thd,
        )[:max_after_nms]
        copied_item["predictions"] = [[video_idx, pred[0], pred[1], pred[2]] for pred in temporal_after_nms]
        processed.append(copied_item)
    return processed


def get_submission_top_n(submission, top_n=100):
    """Keep top-N predictions for each task without mutating the input."""

    def get_prediction_top_n(list_dict_predictions, top_n_):
        top_n_res = []
        for item in list_dict_predictions:
            copied_item = dict(item)
            copied_item["predictions"] = item["predictions"][:top_n_]
            top_n_res.append(copied_item)
        return top_n_res

    top_n_submission = {"video2idx": submission["video2idx"]}
    for task_name in submission:
        if task_name != "video2idx":
            top_n_submission[task_name] = get_prediction_top_n(submission[task_name], top_n)
    return top_n_submission


def compute_context_info(model, eval_dataset, opt):
    """Encode video contexts once and cache them for all query batches."""
    model.eval()
    eval_dataset.set_data_mode("context")
    context_dataloader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_context_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )

    metas = []
    video_feat, video_mask = [], []

    for _, batch in tqdm(
        enumerate(context_dataloader),
        desc="Computing query2video scores",
        total=len(context_dataloader),
    ):
        metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)
        encoded_video = model.encode_context(
            model_inputs["video_feat"],
            model_inputs["video_mask"],
        )
        if "video" in opt.ctx_mode:
            video_feat.append(encoded_video)
            video_mask.append(model_inputs["video_mask"])

    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None

        seq_l = [e.shape[1] for e in tensor_list]
        b_sizes = [e.shape[0] for e in tensor_list]
        b_sizes_cumsum = np.cumsum([0] + b_sizes)

        if len(tensor_list[0].shape) == 3:
            hsz = tensor_list[0].shape[2]
            res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
        elif len(tensor_list[0].shape) == 2:
            res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
        else:
            raise ValueError("Only 2D/3D tensors are supported.")

        for i, tensor_item in enumerate(tensor_list):
            res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i + 1], :seq_l[i]] = tensor_item
        return res_tensor

    return {
        "video_metas": metas,
        "video_feat": cat_tensor(video_feat),
        "video_mask": cat_tensor(video_mask),
    }


def index_if_not_none(input_tensor, indices):
    """Index a tensor only when it exists."""
    if input_tensor is None:
        return input_tensor
    return input_tensor[indices]


def compute_query2ctx_info_svmr_only(model, eval_dataset, opt, ctx_info, max_before_nms=1000):
    """Run SVMR-only inference where each query is evaluated on its ground-truth video."""
    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(True)
    query_eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_query_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )

    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz
    ctx_len = eval_dataset.max_ctx_len

    svmr_video2meta_idx = {entry["vid_name"]: idx for idx, entry in enumerate(video_metas)}
    svmr_gt_st_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)
    svmr_gt_ed_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)

    query_metas = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        batch_query_metas = batch[0]
        query_metas.extend(batch_query_metas)
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)

        query2video_meta_indices = torch.tensor(
            [svmr_video2meta_idx[item["vid_name"]] for item in batch_query_metas],
            dtype=torch.long,
            device=opt.device,
            requires_grad=False,
        )

        _, st_probs, ed_probs = model.get_pred_from_raw_query(
            model_inputs["query_feat"],
            model_inputs["query_mask"],
            index_if_not_none(ctx_info["video_feat"], query2video_meta_indices),
            index_if_not_none(ctx_info["video_mask"], query2video_meta_indices),
            cross=False,
        )

        st_probs = F.softmax(st_probs, dim=-1)
        ed_probs = F.softmax(ed_probs, dim=-1)

        svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :st_probs.shape[1]] = st_probs.cpu().numpy()
        svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :ed_probs.shape[1]] = ed_probs.cpu().numpy()

        if opt.debug:
            break

    n_processed_query = len(query_metas)
    svmr_gt_st_probs = svmr_gt_st_probs[:n_processed_query]
    svmr_gt_ed_probs = svmr_gt_ed_probs[:n_processed_query]
    svmr_res = get_svmr_res_from_st_ed_probs(
        svmr_gt_st_probs,
        svmr_gt_ed_probs,
        query_metas,
        video2idx,
        clip_length=opt.clip_length,
        min_pred_l=opt.min_pred_l,
        max_pred_l=opt.max_pred_l,
        max_before_nms=max_before_nms,
    )
    return {"SVMR": svmr_res}


def generate_min_max_length_mask(array_shape, min_l, max_l):
    """Build a mask for valid (start, end) pairs constrained by span length."""
    single_dims = (1,) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    mask = np.ones(mask_shape, dtype=np.float32)
    mask_triu = np.triu(mask, k=min_l)
    mask_triu_reversed = 1 - np.triu(mask, k=max_l)
    return mask_triu * mask_triu_reversed


def get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx, clip_length,
                                  min_pred_l, max_pred_l, max_before_nms):
    """Convert start/end probabilities into ranked SVMR predictions."""
    svmr_res = []
    query_vid_names = [entry["vid_name"] for entry in query_metas]

    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)
    valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(
        st_ed_prob_product,
        top_n=max_before_nms,
        prob_thd=None,
    )

    for i, query_vid_name in tqdm(
        enumerate(query_vid_names),
        desc="[SVMR] Loop over queries to generate predictions",
        total=len(query_vid_names),
    ):
        query_meta = query_metas[i]
        video_idx = video2idx[query_vid_name]
        sorted_triples = batched_sorted_triples[i]
        sorted_triples[:, :2] = sorted_triples[:, :2] * clip_length
        ranked_predictions = [[video_idx] + row for row in sorted_triples.tolist()]
        svmr_res.append({
            "desc_id": query_meta["desc_id"],
            "desc": query_meta["desc"],
            "predictions": ranked_predictions,
        })
    return svmr_res


def compute_query2ctx_info(model, eval_dataset, opt, ctx_info, max_before_nms=1000, max_n_videos=100,
                           tasks=("SVMR",)):
    """Run query-to-context inference for SVMR/VCMR/VR tasks."""
    is_svmr = "SVMR" in tasks
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks

    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]

    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(is_svmr)

    query_eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_query_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )

    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz
    vcmr_ctx_l = ctx_info["video_feat"].shape[1] if is_vcmr else None

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=np.int32)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)
    else:
        flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores = None, None

    if is_vr or is_vcmr:
        sorted_q2c_indices = np.empty((n_total_query, max_n_videos), dtype=np.int32)
        sorted_q2c_scores = np.empty((n_total_query, max_n_videos), dtype=np.float32)
    else:
        sorted_q2c_indices, sorted_q2c_scores = None, None

    if is_svmr:
        svmr_video2meta_idx = {entry["vid_name"]: idx for idx, entry in enumerate(video_metas)}
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
    else:
        svmr_video2meta_idx, svmr_gt_st_probs, svmr_gt_ed_probs = None, None, None

    query_metas = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        batch_query_metas = batch[0]
        query_metas.extend(batch_query_metas)
        model_inputs = prepare_batch_inputs(batch[1], device=opt.device, non_blocking=opt.pin_memory)

        query_context_scores, st_probs, ed_probs = model.get_pred_from_raw_query(
            model_inputs["query_feat"],
            model_inputs["query_mask"],
            ctx_info["video_feat"],
            ctx_info["video_mask"],
            cross=True,
        )

        query_context_scores = torch.exp(opt.q2c_alpha * query_context_scores)
        st_probs = F.softmax(st_probs, dim=-1)
        ed_probs = F.softmax(ed_probs, dim=-1)

        if is_svmr:
            row_indices = torch.arange(0, len(st_probs), device=st_probs.device)
            query2video_meta_indices = torch.tensor(
                [svmr_video2meta_idx[item["vid_name"]] for item in batch_query_metas],
                dtype=torch.long,
                device=st_probs.device,
            )
            svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :st_probs.shape[2]] = \
                st_probs[row_indices, query2video_meta_indices].cpu().numpy()
            svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :ed_probs.shape[2]] = \
                ed_probs[row_indices, query2video_meta_indices].cpu().numpy()

        if not (is_vr or is_vcmr):
            continue

        sorted_scores, sorted_indices = torch.topk(query_context_scores, max_n_videos, dim=1, largest=True)
        sorted_q2c_indices[idx * bsz:(idx + 1) * bsz] = sorted_indices.cpu().numpy()
        sorted_q2c_scores[idx * bsz:(idx + 1) * bsz] = sorted_scores.cpu().numpy()

        if not is_vcmr:
            continue

        row_indices = torch.arange(0, len(st_probs), device=opt.device).unsqueeze(1)
        st_probs_topk = st_probs[row_indices, sorted_indices]
        ed_probs_topk = ed_probs[row_indices, sorted_indices]

        st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", st_probs_topk, sorted_scores, ed_probs_topk)
        valid_prob_mask = generate_min_max_length_mask(st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
        st_ed_scores *= torch.from_numpy(valid_prob_mask).to(st_ed_scores.device)

        n_q = st_ed_scores.shape[0]
        flat_st_ed_scores = st_ed_scores.reshape(n_q, -1)
        flat_scores, flat_indices = torch.sort(flat_st_ed_scores, dim=1, descending=True)

        flat_st_ed_sorted_scores[idx * bsz:(idx + 1) * bsz] = flat_scores[:, :max_before_nms].cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = flat_indices[:, :max_before_nms].cpu().numpy()

        if opt.debug:
            break

    n_processed_query = len(query_metas)
    if is_svmr:
        svmr_gt_st_probs = svmr_gt_st_probs[:n_processed_query]
        svmr_gt_ed_probs = svmr_gt_ed_probs[:n_processed_query]
    if is_vr or is_vcmr:
        sorted_q2c_indices = sorted_q2c_indices[:n_processed_query]
        sorted_q2c_scores = sorted_q2c_scores[:n_processed_query]
    if is_vcmr:
        flat_st_ed_scores_sorted_indices = flat_st_ed_scores_sorted_indices[:n_processed_query]
        flat_st_ed_sorted_scores = flat_st_ed_sorted_scores[:n_processed_query]
    n_total_query = n_processed_query

    svmr_res = []
    if is_svmr:
        svmr_res = get_svmr_res_from_st_ed_probs(
            svmr_gt_st_probs,
            svmr_gt_ed_probs,
            query_metas,
            video2idx,
            clip_length=opt.clip_length,
            min_pred_l=opt.min_pred_l,
            max_pred_l=opt.max_pred_l,
            max_before_nms=max_before_nms,
        )

    vr_res = []
    if is_vr:
        for i, (scores_row, indices_row) in tqdm(
            enumerate(zip(sorted_q2c_scores[:, :100], sorted_q2c_indices[:, :100])),
            desc="[VR] Loop over queries to generate predictions",
            total=n_total_query,
        ):
            cur_vr_predictions = []
            for score, meta_idx in zip(scores_row, indices_row):
                video_idx = video2idx[video_metas[meta_idx]["vid_name"]]
                cur_vr_predictions.append([video_idx, 0, 0, float(score)])
            vr_res.append({
                "desc_id": query_metas[i]["desc_id"],
                "desc": query_metas[i]["desc"],
                "predictions": cur_vr_predictions,
            })

    vcmr_res = []
    if is_vcmr:
        for i, (flat_indices_row, flat_scores_row) in tqdm(
            enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),
            desc="[VCMR] Loop over queries to generate predictions",
            total=n_total_query,
        ):
            video_meta_indices_local, pred_st_indices, pred_ed_indices = np.unravel_index(
                flat_indices_row,
                shape=(max_n_videos, vcmr_ctx_l, vcmr_ctx_l),
            )
            video_meta_indices = sorted_q2c_indices[i, video_meta_indices_local]
            pred_st_in_seconds = pred_st_indices.astype(np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * opt.clip_length

            cur_vcmr_predictions = []
            for j, (meta_idx, score) in enumerate(zip(video_meta_indices, flat_scores_row)):
                video_idx = video2idx[video_metas[meta_idx]["vid_name"]]
                cur_vcmr_predictions.append([
                    video_idx,
                    float(pred_st_in_seconds[j]),
                    float(pred_ed_in_seconds[j]),
                    float(score),
                ])
            vcmr_res.append({
                "desc_id": query_metas[i]["desc_id"],
                "desc": query_metas[i]["desc"],
                "predictions": cur_vcmr_predictions,
            })

    results = {"SVMR": svmr_res, "VCMR": vcmr_res, "VR": vr_res}
    return {k: v for k, v in results.items() if len(v) != 0}


def get_eval_res(model, eval_dataset, opt, tasks):
    """Compute retrieval predictions for requested tasks."""
    context_info = compute_context_info(model, eval_dataset, opt)
    if "VCMR" in tasks or "VR" in tasks:
        logger.info("Inference with full script.")
        eval_res = compute_query2ctx_info(
            model,
            eval_dataset,
            opt,
            context_info,
            max_before_nms=opt.max_before_nms,
            max_n_videos=opt.max_vcmr_video,
            tasks=tasks,
        )
    else:
        logger.info("Inference in SVMR-only mode.")
        eval_res = compute_query2ctx_info_svmr_only(
            model,
            eval_dataset,
            opt,
            context_info,
            max_before_nms=opt.max_before_nms,
        )
    eval_res["video2idx"] = eval_dataset.video2idx
    return eval_res


POST_PROCESSING_MMS_FUNC = {"SVMR": post_processing_svmr_nms, "VCMR": post_processing_vcmr_nms}


def eval_epoch(model, eval_dataset, opt, save_submission_filename, tasks=("SVMR",), max_after_nms=100):
    """Run one evaluation epoch and optionally evaluate NMS output."""
    model.eval()
    logger.info("Computing scores")
    st_time = time.time()
    eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks)
    total_time = time.time() - st_time
    print("\n" + "\x1b[1;31m" + str(total_time) + "\x1b[0m", flush=True)

    iou_thds = (0.5, 0.7)
    logger.info("Saving and evaluating raw results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    has_gt = opt.dset_name != "tvr" or opt.eval_split_name == "val"
    if has_gt:
        metrics = eval_retrieval(
            eval_submission,
            eval_dataset.query_data,
            iou_thds=iou_thds,
            match_number=not opt.debug,
            verbose=opt.debug,
            use_desc_type=opt.dset_name == "tvr",
        )
        save_metrics_path = submission_path.replace(".json", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path]

    if opt.nms_thd != -1:
        logger.info("Performing NMS with threshold %s", opt.nms_thd)
        eval_submission_after_nms = {"video2idx": eval_submission_raw["video2idx"]}
        for task_name, nms_func in POST_PROCESSING_MMS_FUNC.items():
            if task_name in eval_submission_raw:
                eval_submission_after_nms[task_name] = nms_func(
                    eval_submission_raw[task_name],
                    nms_thd=opt.nms_thd,
                    max_before_nms=opt.max_before_nms,
                    max_after_nms=max_after_nms,
                )

        submission_nms_path = submission_path.replace(".json", "_nms_thd_{}.json".format(opt.nms_thd))
        save_json(eval_submission_after_nms, submission_nms_path)

        if has_gt:
            metrics_nms = eval_retrieval(
                eval_submission_after_nms,
                eval_dataset.query_data,
                iou_thds=iou_thds,
                match_number=not opt.debug,
                verbose=opt.debug,
                use_desc_type=opt.dset_name == "tvr",
            )
            save_metrics_nms_path = submission_nms_path.replace(".json", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path]
    else:
        metrics_nms = None

    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load checkpoint and move model to the target device."""
    checkpoint = torch.load(opt.ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    model = ReLoCLNet(loaded_model_cfg)
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model from epoch %s: %s", checkpoint["epoch"], opt.ckpt_filepath)

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU %s", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)
    return model


def start_inference():
    """Entry point for offline inference."""
    logger.info("Setup config, data, and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    assert opt.eval_path is not None
    eval_dataset = StartEndEvalDataset(
        dset_name=opt.dset_name,
        eval_split_name=opt.eval_split_name,
        data_path=opt.eval_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_duration_idx_path=opt.video_duration_idx_path,
        vid_feat_path_or_handler=opt.vid_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        data_mode="query",
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
    )

    model = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_predictions_{}.json".format(
        opt.dset_name,
        opt.eval_split_name,
        opt.eval_id,
        "_".join(opt.tasks),
    )
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, _ = eval_epoch(
            model,
            eval_dataset,
            opt,
            save_submission_filename,
            tasks=opt.tasks,
            max_after_nms=100,
        )
    logger.info("metrics_no_nms\n%s", pprint.pformat(metrics_no_nms, indent=4))
    logger.info("metrics_nms\n%s", pprint.pformat(metrics_nms, indent=4))


if __name__ == '__main__':
    start_inference()
