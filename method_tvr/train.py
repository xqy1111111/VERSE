import os
import sys
import time
import json
import pprint
import random
import numpy as np
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from method_tvr.config import BaseOptions
from method_tvr.semantic_perturb.cache_builder import SemanticBuildConfig, build_semantic_cache
from method_tvr.semantic_perturb.dataset_semantic import SemanticLossRuntime, load_semantic_cache_lookup
from method_tvr.model import ReLoCLNet
from method_tvr.start_end_dataset import StartEndDataset, start_end_collate, StartEndEvalDataset, prepare_batch_inputs
from method_tvr.inference import eval_epoch, start_inference
from method_tvr.optimization import BertAdam
from utils.basic_utils import AverageMeter, save_json
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def _safe_metric_sum(task_metrics, metric_names):
    return float(sum(float(task_metrics.get(metric_name, 0.0)) for metric_name in metric_names))


def _get_early_stop_metric_names(opt, task_name):
    task_to_opt = {
        "VCMR": ("early_stop_vcmr_metrics", ["0.5-r1", "0.7-r1"]),
        "SVMR": ("early_stop_svmr_metrics", ["0.5-r1", "0.7-r1"]),
        "VR": ("early_stop_vr_metrics", ["r1"]),
    }
    attr_name, default_metrics = task_to_opt[task_name]
    metric_names = getattr(opt, attr_name, default_metrics)
    if metric_names is None:
        return list(default_metrics)
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    metric_names = [str(metric_name).strip() for metric_name in metric_names if str(metric_name).strip()]
    return metric_names if metric_names else list(default_metrics)


def _compute_stop_score(metrics, opt):
    if getattr(opt, "early_stop_use_composite", False):
        vcmr_metric_names = _get_early_stop_metric_names(opt, "VCMR")
        svmr_metric_names = _get_early_stop_metric_names(opt, "SVMR")
        vr_metric_names = _get_early_stop_metric_names(opt, "VR")

        vcmr_score = _safe_metric_sum(metrics.get("VCMR", {}), vcmr_metric_names)
        svmr_score = _safe_metric_sum(metrics.get("SVMR", {}), svmr_metric_names)
        vr_score = _safe_metric_sum(metrics.get("VR", {}), vr_metric_names)
        if getattr(opt, "early_stop_normalize_components", False):
            vcmr_score = vcmr_score / float(opt.early_stop_vcmr_scale)
            svmr_score = svmr_score / float(opt.early_stop_svmr_scale)
            vr_score = vr_score / float(opt.early_stop_vr_scale)
        stop_score = (
            opt.early_stop_vcmr_weight * vcmr_score
            + opt.early_stop_svmr_weight * svmr_score
            + opt.early_stop_vr_weight * vr_score
        )
        if getattr(opt, "early_stop_normalize_components", False):
            stop_desc = (
                "composite_norm("
                f"vcmr/{opt.early_stop_vcmr_scale},"
                f"svmr/{opt.early_stop_svmr_scale},"
                f"vr/{opt.early_stop_vr_scale})"
            )
        else:
            stop_desc = "composite"
        return stop_score, stop_desc

    stop_metric_names = _get_early_stop_metric_names(opt, opt.stop_task)
    stop_score = _safe_metric_sum(metrics.get(opt.stop_task, {}), stop_metric_names)
    stop_desc = " ".join([opt.stop_task] + stop_metric_names)
    return stop_score, stop_desc


def _extract_semantic_opt_dict(opt):
    keys = [
        "enable_compositional_supervision",
        "positive_rewrite_sample_size",
        "negative_rewrite_sample_size",
        "positive_invariance_weight",
        "negative_preference_weight",
        "enable_debiased_retrieval_correction",
        "debiased_retrieval_weight",
        "compositional_warmup_epochs",
        "compositional_ramp_epochs",
        "negative_preference_delay_epochs",
        "debiased_retrieval_delay_epochs",
        "rewrite_type_quota_enabled",
        "risky_negative_filter_enabled",
        "risky_negative_overlap_threshold",
        "risky_negative_start_epoch",
        "risky_negative_downweight",
        "collision_sanitization_enabled",
        "allow_missing_rewrites",
        "require_rewrite_cache",
        "semantic_enable",
        "semantic_backend",
        "semantic_strict_mode",
        "semantic_no_fallback",
        "semantic_cache_path",
        "semantic_fail_on_missing_cache",
        "semantic_fail_on_invalid_cache",
        "semantic_cache_split",
        "semantic_num_hard_neg",
        "semantic_num_hard_pos",
        "semantic_max_retries_same_backend",
        "semantic_prompt_version",
        "semantic_schema_version",
        "semantic_generator_model",
        "semantic_verifier_model",
        "semantic_temperature",
        "semantic_seed",
        "semantic_llm_api_base",
        "semantic_llm_api_key",
        "semantic_llm_transport",
        "semantic_llm_response_mode",
        "semantic_local_model_name_or_path",
        "semantic_local_device",
        "semantic_local_mask_backend",
        "semantic_local_max_new_tokens",
        "semantic_neg_types",
        "semantic_pos_types",
        "semantic_severity_levels",
        "semantic_use_preference_loss",
        "semantic_preference_margin",
        "semantic_preference_weight",
        "semantic_use_consistency_loss",
        "semantic_consistency_weight",
        "semantic_text_encoder_name_or_path",
    ]
    return {k: getattr(opt, k) for k in keys}


def _build_semantic_cache_from_opt(opt):
    cfg = SemanticBuildConfig(
        dset_name=opt.dset_name,
        source_path=opt.train_path,
        cache_split=opt.semantic_cache_split,
        output_path=opt.semantic_cache_path,
        backend=opt.semantic_backend,
        strict_mode=opt.semantic_strict_mode,
        no_fallback=opt.semantic_no_fallback,
        num_hard_neg=opt.semantic_num_hard_neg,
        num_hard_pos=opt.semantic_num_hard_pos,
        max_retries_same_backend=opt.semantic_max_retries_same_backend,
        prompt_version=opt.semantic_prompt_version,
        schema_version=opt.semantic_schema_version,
        generator_model=opt.semantic_generator_model,
        verifier_model=opt.semantic_verifier_model,
        temperature=opt.semantic_temperature,
        seed=opt.semantic_seed,
        neg_types=opt.semantic_neg_types,
        pos_types=opt.semantic_pos_types,
        severity_levels=opt.semantic_severity_levels,
        llm_api_base=opt.semantic_llm_api_base,
        llm_api_key=opt.semantic_llm_api_key,
        llm_transport=opt.semantic_llm_transport,
        llm_response_mode=opt.semantic_llm_response_mode,
        local_model_name_or_path=opt.semantic_local_model_name_or_path,
        local_device=opt.semantic_local_device,
        local_mask_backend=opt.semantic_local_mask_backend,
        local_max_new_tokens=opt.semantic_local_max_new_tokens,
    )
    return build_semantic_cache(cfg)


def train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True, semantic_runtime=None):
    logger.info("use train_epoch func for training: {}".format(training))
    model.train(mode=training)
    model.set_train_epoch(epoch_i)
    if semantic_runtime is not None:
        semantic_runtime.set_current_epoch(epoch_i)
        schedule = semantic_runtime.get_schedule_snapshot()
        logger.info(
            "compositional_supervision=%s schedule(pos=%.3f neg=%.3f deb=%.3f) sample_size(pos=%d neg=%d) quota=%s collision_sanitize=%s risky_filter=%s threshold=%.2f",
            bool(getattr(opt, "enable_compositional_supervision", False) or getattr(opt, "semantic_enable", False)),
            schedule.get("compositional_schedule_positive", 0.0),
            schedule.get("compositional_schedule_negative", 0.0),
            schedule.get("compositional_schedule_debiased", 0.0),
            int(getattr(opt, "positive_rewrite_sample_size", getattr(opt, "semantic_num_hard_pos", 0))),
            int(getattr(opt, "negative_rewrite_sample_size", getattr(opt, "semantic_num_hard_neg", 0))),
            bool(getattr(opt, "rewrite_type_quota_enabled", False)),
            bool(getattr(opt, "collision_sanitization_enabled", False)),
            bool(getattr(opt, "risky_negative_filter_enabled", False)),
            float(getattr(opt, "risky_negative_overlap_threshold", 0.0)),
        )
    model_core_for_log = model.module if isinstance(model, torch.nn.DataParallel) else model
    logger.info(
        ("epoch %d retrieval_scorer=%s late_topk(train/eval/default)=%s/%s/%s "
         "late_q_chunk=%s late_margin_gate=%.4f late_weight(train/eval/default)=%.4f/%.4f/%.4f "
         "late_start_epoch=%d late_clip=%.4f late_soft_gate(T/min)=%.4f/%.4f "
         "late_warmup=%d late_detach_backbone=%s late_vcl=%s"),
        epoch_i,
        getattr(model_core_for_log.config, "retrieval_scorer", "single_vector"),
        getattr(model_core_for_log.config, "late_interaction_train_rerank_topk", 0),
        getattr(model_core_for_log.config, "late_interaction_eval_rerank_topk", 0),
        getattr(model_core_for_log.config, "late_interaction_rerank_topk", 0),
        getattr(model_core_for_log.config, "late_interaction_query_chunk_size", 0),
        float(getattr(model_core_for_log.config, "late_interaction_rerank_margin_threshold", -1.0)),
        float(getattr(model_core_for_log.config, "late_interaction_train_score_weight", 0.0)),
        float(getattr(model_core_for_log.config, "late_interaction_eval_score_weight", 0.0)),
        float(getattr(model_core_for_log.config, "late_interaction_score_weight", 0.0)),
        int(getattr(model_core_for_log.config, "late_interaction_train_start_epoch", 0)),
        float(getattr(model_core_for_log.config, "late_interaction_residual_clip", -1.0)),
        float(getattr(model_core_for_log.config, "late_interaction_rerank_soft_temperature", 0.0)),
        float(getattr(model_core_for_log.config, "late_interaction_rerank_soft_min_gate", 0.0)),
        int(getattr(model_core_for_log.config, "late_interaction_train_score_warmup_epochs", 0)),
        bool(getattr(model_core_for_log.config, "late_interaction_detach_backbone_in_train", False)),
        bool(getattr(model_core_for_log.config, "late_interaction_apply_to_vcl", False)),
    )
    if opt.hard_negative_start_epoch != -1 and epoch_i >= opt.hard_negative_start_epoch:
        model.set_hard_negative(True, opt.hard_pool_size)
    if opt.train_span_start_epoch != -1 and epoch_i >= opt.train_span_start_epoch:
        model.set_train_st_ed(opt.lw_st_ed)
        model.set_train_span_joint(opt.lw_span_joint)

    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meters = OrderedDict(loss_st_ed=AverageMeter(),
                              loss_span_joint=AverageMeter(),
                              loss_fcl=AverageMeter(), loss_vcl=AverageMeter(),
                              loss_debiased_video_frame=AverageMeter(),
                              loss_neg_ctx=AverageMeter(), loss_neg_q=AverageMeter(), loss_lm=AverageMeter(),
                              loss_overall=AverageMeter())
    if getattr(opt, "semantic_enable", False):
        loss_meters["loss_semantic_pref"] = AverageMeter()
        loss_meters["loss_semantic_cons"] = AverageMeter()
        loss_meters["loss_semantic_debiased"] = AverageMeter()
        loss_meters["loss_semantic_total"] = AverageMeter()

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
        global_step = epoch_i * num_training_examples + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)

        # continue
        use_semantic = bool(getattr(opt, "semantic_enable", False) and training and semantic_runtime is not None)
        timer_start = time.time()
        batch_meta = batch[0]
        model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)
        timer_start = time.time()
        model_aux = None
        if use_semantic:
            loss, loss_dict, model_aux = model(**model_inputs, return_aux=True)
        else:
            loss, loss_dict = model(**model_inputs)

        if use_semantic:
            model_core = model.module if isinstance(model, torch.nn.DataParallel) else model
            loss_semantic, loss_semantic_dict = semantic_runtime.compute_losses(
                model_core=model_core,
                model_inputs=model_inputs,
                batch_meta=batch_meta,
                model_aux=model_aux,
            )
            loss = loss + loss_semantic
            loss_dict.update(loss_semantic_dict)
            loss_dict["loss_overall"] = float(loss.detach().cpu().item())
        model_forward_time.update(time.time() - timer_start)
        timer_start = time.time()
        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)

            opt.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            for k, v in loss_dict.items():
                opt.writer.add_scalar("Train/{}".format(k), v, global_step)

        for k, v in loss_dict.items():
            if k not in loss_meters:
                loss_meters[k] = AverageMeter()
            loss_meters[k].update(float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    if training:
        to_write = opt.train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch_i,
                                                      loss_str=" ".join(["{} {:.4f}".format(k, v.avg)
                                                                         for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n".format(
            dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
            model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        for k, v in loss_meters.items():
            opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)


def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def train(model, train_dataset, train_eval_dataset, val_dataset, opt, semantic_runtime=None):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    train_loader = DataLoader(train_dataset, collate_fn=start_end_collate, batch_size=opt.bsz,
                              num_workers=opt.num_workers, shuffle=True, pin_memory=opt.pin_memory)
    train_eval_loader = None
    if train_eval_dataset is not None:
        train_eval_loader = DataLoader(train_eval_dataset, collate_fn=start_end_collate, batch_size=opt.bsz,
                                       num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    num_train_optimization_steps = len(train_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd, warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps, schedule="warmup_linear")
    prev_best_score = float("-inf")
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0
    eval_tasks_at_training = opt.eval_tasks_at_training  # VR is computed along with VCMR
    save_submission_filename = "latest_{}_{}_predictions_{}.json".format(opt.dset_name, opt.eval_split_name,
                                                                         "_".join(eval_tasks_at_training))
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            if opt.debug:
                with torch.autograd.detect_anomaly():
                    train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True, semantic_runtime=semantic_runtime)
            else:
                train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True, semantic_runtime=semantic_runtime)
        global_step = (epoch_i + 1) * len(train_loader)
        if opt.eval_path is not None and train_eval_loader is not None:
            with torch.no_grad():
                train_epoch(model, train_eval_loader, optimizer, opt, epoch_i, training=False, semantic_runtime=semantic_runtime)
                metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch(
                    model, val_dataset, opt, save_submission_filename, tasks=eval_tasks_at_training, max_after_nms=100)
            to_write = opt.eval_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch_i,
                                                         eval_metrics_str=json.dumps(metrics_no_nms))
            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(
                rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
            logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms, indent=4)))
            # metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
            metrics = metrics_no_nms
            # early stop/ log / save model
            for task_type in ["SVMR", "VCMR"]:
                if task_type in metrics:
                    task_metrics = metrics[task_type]
                    for iou_thd in [0.5, 0.7]:
                        opt.writer.add_scalars("Eval/{}-{}".format(task_type, iou_thd),
                                               {k: v for k, v in task_metrics.items() if str(iou_thd) in k},
                                               global_step)
            task_type = "VR"
            if task_type in metrics:
                task_metrics = metrics[task_type]
                opt.writer.add_scalars("Eval/{}".format(task_type), {k: v for k, v in task_metrics.items()},
                                       global_step)
            stop_score, stop_desc = _compute_stop_score(metrics, opt)
            if stop_score > prev_best_score + opt.early_stop_min_delta:
                es_cnt = 0
                prev_best_score = stop_score
                checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
                torch.save(checkpoint, opt.ckpt_filepath)
                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt >= opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write("Early Stop at epoch {}".format(epoch_i))
                    logger.info("Early stop at {} with {} {}".format(
                        epoch_i, stop_desc, prev_best_score))
                    break
        else:
            checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

        if opt.debug:
            break

    opt.writer.close()


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    logger.info("retrieval_scorer=%s", opt.retrieval_scorer)
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    if getattr(opt, "semantic_build_cache_only", False):
        out_paths = _build_semantic_cache_from_opt(opt)
        save_json(
            {
                "semantic_config": _extract_semantic_opt_dict(opt),
                "outputs": out_paths,
            },
            os.path.join(opt.results_dir, "semantic_cache_build_outputs.json"),
            save_pretty=True,
            sort_keys=True,
        )
        logger.info("Semantic cache build-only completed: {}".format(out_paths))
        return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug, True

    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    tokenizer = None
    if opt.use_generative_augmentation:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_name_or_path, use_fast=True)
        if not opt.lm_disable_start_token:
            if opt.lm_start_token_id is None:
                if opt.lm_start_token not in tokenizer.get_vocab():
                    tokenizer.add_special_tokens({"additional_special_tokens": [opt.lm_start_token]})
                opt.lm_start_token_id = tokenizer.convert_tokens_to_ids(opt.lm_start_token)
            elif opt.lm_start_token_id >= len(tokenizer):
                raise ValueError("--lm_start_token_id is outside tokenizer vocab size.")
        if opt.lm_vocab_size is None or opt.lm_vocab_size < len(tokenizer):
            opt.lm_vocab_size = len(tokenizer)
        if opt.lm_pad_token_id is None:
            opt.lm_pad_token_id = tokenizer.pad_token_id
            if opt.lm_pad_token_id is None:
                opt.lm_pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    semantic_cache_lookup = None
    semantic_runtime = None
    if getattr(opt, "semantic_enable", False):
        semantic_cache_lookup = load_semantic_cache_lookup(opt, source_data_path=opt.train_path)
        semantic_runtime = SemanticLossRuntime(opt)
        if semantic_cache_lookup is None:
            logger.warning(
                "compositional supervision enabled but no rewrite cache is loaded; training will automatically fall back to base objective for missing rewrites."
            )
        semantic_summary = {
            "semantic_config": _extract_semantic_opt_dict(opt),
            "cache_manifest": semantic_cache_lookup.manifest if semantic_cache_lookup is not None else None,
            "cache_path": opt.semantic_cache_path,
        }
        save_json(semantic_summary, os.path.join(opt.results_dir, "semantic_summary.json"), save_pretty=True, sort_keys=True)

    train_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        vid_feat_path_or_handler=opt.vid_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat,
        tokenizer=tokenizer,
        lm_max_len=opt.lm_max_len,
        lm_start_token_id=opt.lm_start_token_id,
        semantic_cache_lookup=semantic_cache_lookup)

    if opt.eval_path is not None:
        # val dataset, used to get eval loss
        train_eval_dataset = StartEndDataset(
            dset_name=opt.dset_name,
            data_path=opt.eval_path,
            desc_bert_path_or_handler=train_dataset.desc_bert_h5,
            max_desc_len=opt.max_desc_l,
            max_ctx_len=opt.max_ctx_l,
            vid_feat_path_or_handler=train_dataset.vid_feat_h5 if "video" in opt.ctx_mode else None,
            clip_length=opt.clip_length,
            ctx_mode=opt.ctx_mode,
            h5driver=opt.h5driver,
            data_ratio=opt.data_ratio,
            normalize_vfeat=not opt.no_norm_vfeat,
            normalize_tfeat=not opt.no_norm_tfeat,
            tokenizer=tokenizer,
            lm_max_len=opt.lm_max_len,
            lm_start_token_id=opt.lm_start_token_id)

        eval_dataset = StartEndEvalDataset(
            dset_name=opt.dset_name,
            eval_split_name=opt.eval_split_name,  # should only be val set
            data_path=opt.eval_path,
            desc_bert_path_or_handler=train_dataset.desc_bert_h5,
            max_desc_len=opt.max_desc_l,
            max_ctx_len=opt.max_ctx_l,
            video_duration_idx_path=opt.video_duration_idx_path,
            vid_feat_path_or_handler=train_dataset.vid_feat_h5 if "video" in opt.ctx_mode else None,
            clip_length=opt.clip_length,
            ctx_mode=opt.ctx_mode,
            data_mode="query",
            h5driver=opt.h5driver,
            data_ratio=opt.data_ratio,
            normalize_vfeat=not opt.no_norm_vfeat,
            normalize_tfeat=not opt.no_norm_tfeat)
    else:
        train_eval_dataset, eval_dataset = None, None

    model_config = EDict(
        visual_input_size=opt.vid_feat_size,
        query_input_size=opt.q_feat_size,
        hidden_size=opt.hidden_size,  # hidden dimension
        conv_kernel_size=opt.conv_kernel_size,
        conv_stride=opt.conv_stride,
        span_head_type=opt.span_head_type,
        span_biaffine_hidden_size=opt.span_biaffine_hidden_size,
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        input_drop=opt.input_drop,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        ctx_mode=opt.ctx_mode,
        backbone_type=opt.backbone_type,
        retrieval_scorer=opt.retrieval_scorer,
        late_interaction_dim=opt.late_interaction_dim,
        late_interaction_use_projection=opt.late_interaction_use_projection,
        late_interaction_use_token_weight=opt.late_interaction_use_token_weight,
        late_interaction_token_weight_floor=opt.late_interaction_token_weight_floor,
        late_interaction_score_reduction=opt.late_interaction_score_reduction,
        late_interaction_video_chunk_size=opt.late_interaction_video_chunk_size,
        late_interaction_rerank_topk=opt.late_interaction_rerank_topk,
        late_interaction_train_rerank_topk=opt.late_interaction_train_rerank_topk,
        late_interaction_eval_rerank_topk=opt.late_interaction_eval_rerank_topk,
        late_interaction_query_chunk_size=opt.late_interaction_query_chunk_size,
        late_interaction_rerank_margin_threshold=opt.late_interaction_rerank_margin_threshold,
        late_interaction_rerank_soft_temperature=opt.late_interaction_rerank_soft_temperature,
        late_interaction_rerank_soft_min_gate=opt.late_interaction_rerank_soft_min_gate,
        late_interaction_train_start_epoch=opt.late_interaction_train_start_epoch,
        late_interaction_train_score_weight=opt.late_interaction_train_score_weight,
        late_interaction_eval_score_weight=opt.late_interaction_eval_score_weight,
        late_interaction_train_score_warmup_epochs=opt.late_interaction_train_score_warmup_epochs,
        late_interaction_residual_clip=opt.late_interaction_residual_clip,
        late_interaction_detach_backbone_in_train=opt.late_interaction_detach_backbone_in_train,
        late_interaction_score_weight=opt.late_interaction_score_weight,
        late_interaction_score_normalize=opt.late_interaction_score_normalize,
        late_interaction_apply_to_vcl=opt.late_interaction_apply_to_vcl,
        multi_vector_query_max_count=opt.multi_vector_query_max_count,
        multi_vector_phrase_window=opt.multi_vector_phrase_window,
        multi_vector_use_phrase_pooling=opt.multi_vector_use_phrase_pooling,
        multi_vector_use_global_fallback=opt.multi_vector_use_global_fallback,
        event_token_compression_enabled=opt.event_token_compression_enabled,
        event_token_compression_keep_ratio=opt.event_token_compression_keep_ratio,
        event_token_compression_min_tokens=opt.event_token_compression_min_tokens,
        event_token_compression_max_tokens=opt.event_token_compression_max_tokens,
        event_token_compression_add_event_token=opt.event_token_compression_add_event_token,
        event_token_compression_temperature=opt.event_token_compression_temperature,
        event_token_compression_anchor_mode=opt.event_token_compression_anchor_mode,
        use_generative_augmentation=opt.use_generative_augmentation,
        use_fusion_encoder=opt.use_fusion_encoder,
        fusion_num_layers=opt.fusion_num_layers,
        lm_weight=opt.lm_weight,
        lm_vocab_size=opt.lm_vocab_size,
        lm_pad_token_id=opt.lm_pad_token_id,
        lm_num_layers=opt.lm_num_layers,
        mamba_d_state=opt.mamba_d_state,
        mamba_d_conv=opt.mamba_d_conv,
        mamba_expand=opt.mamba_expand,
        mamba_fuse_mode=opt.mamba_fuse_mode,
        margin=opt.margin,  # margin for ranking loss
        ranking_loss_type=opt.ranking_loss_type,  # loss type, 'hinge' or 'lse'
        lw_neg_q=opt.lw_neg_q,  # loss weight for neg. query and pos. context
        lw_neg_ctx=opt.lw_neg_ctx,  # loss weight for pos. query and neg. context
        lw_fcl=opt.lw_fcl,  # loss weight for frame level contrastive learning
        lw_vcl=opt.lw_vcl,  # loss weight for video level contrastive learning
        enable_debiased_video_frame_loss=opt.enable_debiased_video_frame_loss,
        lw_debiased_video_frame_loss=opt.lw_debiased_video_frame_loss,
        debiased_video_frame_start_epoch=opt.debiased_video_frame_start_epoch,
        debiased_video_frame_temperature=opt.debiased_video_frame_temperature,
        debiased_video_frame_gap_threshold=opt.debiased_video_frame_gap_threshold,
        debiased_video_frame_gap_temperature=opt.debiased_video_frame_gap_temperature,
        debiased_video_frame_min_negative_weight=opt.debiased_video_frame_min_negative_weight,
        debiased_video_frame_background_similarity_threshold=opt.debiased_video_frame_background_similarity_threshold,
        debiased_video_frame_background_temperature=opt.debiased_video_frame_background_temperature,
        debiased_video_frame_background_downweight=opt.debiased_video_frame_background_downweight,
        lw_st_ed=0,  # will be assigned dynamically at training time
        lw_span_joint=0,  # will be assigned dynamically at training time
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size)
    logger.info("model_config {}".format(model_config))
    model = ReLoCLNet(model_config)
    count_parameters(model)
    logger.info("Start Training...")
    train(model, train_dataset, train_eval_dataset, eval_dataset, opt, semantic_runtime=semantic_runtime)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug, False


if __name__ == '__main__':
    model_dir, eval_split_name, eval_path, debug, build_only = start_training()
    if not debug and not build_only:
        # Keep absolute run directory so post-train inference can load from any results_root.
        model_dir = os.path.abspath(model_dir)
        tasks = ["SVMR", "VCMR", "VR"]
        input_args = ["--model_dir", model_dir, "--nms_thd", "0.5", "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path, "--tasks"] + tasks
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model in {}".format(model_dir))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference()
