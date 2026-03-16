import logging
import math

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from method_tvr.config import BaseOptions
from utils.basic_utils import l2_normalize_np_array, load_json, load_jsonl, uniform_feature_sampling
from utils.tensor_utils import pad_sequences_1d

logger = logging.getLogger(__name__)


def _normalize_record(raw_data):
    vid_name = raw_data.get("vid_name") or raw_data.get("video")
    ts = raw_data.get("ts") or raw_data.get("time")
    desc = raw_data.get("desc") or raw_data.get("fig_desc") or raw_data.get("cog_desc")
    if vid_name is None or ts is None or desc is None:
        raise ValueError("Missing fields in record: {}".format(raw_data))
    return {
        "desc_id": raw_data["desc_id"],
        "desc": desc,
        "vid_name": vid_name,
        "duration": raw_data["duration"],
        "ts": ts,
    }


class StartEndDataset(Dataset):
    def __init__(
        self,
        dset_name,
        data_path,
        desc_bert_path_or_handler,
        max_desc_len,
        max_ctx_len,
        vid_feat_path_or_handler,
        clip_length,
        ctx_mode="video",
        normalize_vfeat=True,
        normalize_tfeat=True,
        h5driver=None,
        data_ratio=1.0,
        tokenizer=None,
        lm_max_len=30,
        lm_start_token_id=None,
        semantic_cache_lookup=None,
    ):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len
        self.clip_length = clip_length
        self.ctx_mode = ctx_mode

        self.data = [_normalize_record(e) for e in load_jsonl(data_path)]
        if self.data_ratio != 1:
            n_examples = int(len(self.data) * data_ratio)
            self.data = self.data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))

        self.tokenizer = tokenizer
        self.lm_max_len = lm_max_len
        self.lm_start_token_id = lm_start_token_id
        self.semantic_cache_lookup = semantic_cache_lookup

        self.use_video = "video" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode

        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:
                self.vid_feat_h5 = h5py.File(vid_feat_path_or_handler, "r", driver=h5driver)

        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_data = self.data[index]
        meta = {
            "desc_id": raw_data["desc_id"],
            "desc": raw_data["desc"],
            "vid_name": raw_data["vid_name"],
            "duration": raw_data["duration"],
            "ts": raw_data["ts"],
        }
        if self.semantic_cache_lookup is not None:
            semantic_entry = self.semantic_cache_lookup.get_entry(meta["desc_id"])
            if semantic_entry is not None:
                meta["semantic"] = {
                    "hard_negatives": semantic_entry["hard_negatives"],
                    "hard_positives": semantic_entry["hard_positives"],
                }

        model_inputs = {"query_feat": self.get_query_feat_by_desc_id(meta["desc_id"])}

        if self.tokenizer is not None:
            max_len = self.lm_max_len
            if self.lm_start_token_id is not None:
                max_len = max(1, self.lm_max_len - 1)
            encoded = self.tokenizer(
                meta["desc"],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attn_mask = encoded["attention_mask"].squeeze(0)
            if self.lm_start_token_id is not None:
                start_token = input_ids.new_tensor([self.lm_start_token_id])
                input_ids = torch.cat([start_token, input_ids], dim=0)[:self.lm_max_len]
                attn_mask = torch.cat([attn_mask.new_ones(1), attn_mask], dim=0)[:self.lm_max_len]
            model_inputs["query_input_ids"] = input_ids
            model_inputs["query_attn_mask"] = attn_mask

        ctx_l = 0
        if self.use_video:
            video_feat = uniform_feature_sampling(self.vid_feat_h5[meta["vid_name"]][:], self.max_ctx_len)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            video_feat = torch.from_numpy(video_feat)
            ctx_l = len(video_feat)
        else:
            video_feat = torch.zeros((2, 2))

        if self.use_tef:
            ctx_l = int(meta["duration"] // self.clip_length + 1) if ctx_l == 0 else ctx_l
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = torch.arange(1, ctx_l + 1, 1.0) / ctx_l
            tef_feat = torch.stack([tef_st, tef_ed], dim=1)
            if self.use_video:
                video_feat = torch.cat([video_feat, tef_feat], dim=1)
            else:
                video_feat = tef_feat

        model_inputs["video_feat"] = video_feat
        model_inputs["st_ed_indices"] = self.get_st_ed_label(meta["ts"], max_idx=ctx_l - 1)
        return {"meta": meta, "model_inputs": model_inputs}

    def get_st_ed_label(self, ts, max_idx):
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        ed_idx = min(math.ceil(ts[1] / self.clip_length), max_idx)
        return torch.tensor([st_idx, ed_idx], dtype=torch.long)

    def get_query_feat_by_desc_id(self, desc_id):
        query_feat = self.desc_bert_h5[str(desc_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)


class StartEndEvalDataset(Dataset):
    def __init__(
        self,
        dset_name,
        eval_split_name,
        data_path=None,
        desc_bert_path_or_handler=None,
        max_desc_len=None,
        max_ctx_len=None,
        vid_feat_path_or_handler=None,
        video_duration_idx_path=None,
        clip_length=None,
        ctx_mode="video",
        data_mode="context",
        h5driver=None,
        data_ratio=1.0,
        normalize_vfeat=True,
        normalize_tfeat=True,
    ):
        self.dset_name = dset_name
        self.eval_split_name = eval_split_name
        self.ctx_mode = ctx_mode
        self.load_gt_video = False
        self.data_ratio = data_ratio
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat

        self.data_mode = None
        self.set_data_mode(data_mode)

        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len
        self.data_path = data_path

        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        video_data = load_json(video_duration_idx_path)[self.eval_split_name]
        self.video_data = [{"vid_name": k, "duration": v[0]} for k, v in video_data.items()]
        self.video2idx = {k: v[1] for k, v in video_data.items()}
        self.clip_length = clip_length

        self.use_video = "video" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode

        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:
                self.vid_feat_h5 = h5py.File(vid_feat_path_or_handler, "r", driver=h5driver)

        self.query_data = [_normalize_record(e) for e in load_jsonl(data_path)]
        if data_ratio != 1:
            n_examples = int(len(self.query_data) * data_ratio)
            self.query_data = self.query_data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))

    def set_data_mode(self, data_mode):
        assert data_mode in ["context", "query"]
        self.data_mode = data_mode

    def load_gt_vid_name_for_query(self, load_gt_video):
        if load_gt_video:
            assert "vid_name" in self.query_data[0]
        self.load_gt_video = load_gt_video

    def __len__(self):
        if self.data_mode == "context":
            return len(self.video_data)
        return len(self.query_data)

    def __getitem__(self, index):
        if self.data_mode == "context":
            return self._get_item_context(index)
        return self._get_item_query(index)

    def get_query_feat_by_desc_id(self, desc_id):
        query_feat = self.desc_bert_h5[str(desc_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def _get_item_query(self, index):
        raw_data = self.query_data[index]
        meta = {
            "desc_id": raw_data["desc_id"],
            "desc": raw_data["desc"],
            "vid_name": raw_data["vid_name"] if self.load_gt_video else None,
        }
        model_inputs = {"query_feat": self.get_query_feat_by_desc_id(meta["desc_id"])}
        return {"meta": meta, "model_inputs": model_inputs}

    def _get_item_context(self, index):
        raw_data = self.video_data[index]
        meta = {"vid_name": raw_data["vid_name"], "duration": raw_data["duration"]}
        model_inputs = {}
        ctx_l = 0

        if self.use_video:
            video_feat = uniform_feature_sampling(self.vid_feat_h5[meta["vid_name"]][:], self.max_ctx_len)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            video_feat = torch.from_numpy(video_feat)
            ctx_l = len(video_feat)
        else:
            video_feat = torch.zeros((2, 2))

        if self.use_tef:
            ctx_l = int(meta["duration"] // self.clip_length + 1) if ctx_l == 0 else ctx_l
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef_feat = torch.stack([tef_st, tef_ed], dim=1)
            if self.use_video:
                video_feat = torch.cat([video_feat, tef_feat], dim=1)
            else:
                video_feat = tef_feat

        model_inputs["video_feat"] = video_feat
        return {"meta": meta, "model_inputs": model_inputs}


def _get_fixed_length(batch, model_inputs_keys):
    lengths = []
    for key in ("video_feat",):
        if key in model_inputs_keys:
            lengths.extend([e["model_inputs"][key].shape[0] for e in batch])
    if lengths:
        return max(max(lengths), 128)
    return 128


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    fixed_length = _get_fixed_length(batch, model_inputs_keys)
    batched_data = {}

    for key in model_inputs_keys:
        if "feat" in key:
            pad_len = fixed_length if key in ["video_feat"] else None
            batched_data[key] = pad_sequences_1d(
                [e["model_inputs"][key] for e in batch],
                dtype=torch.float32,
                fixed_length=pad_len,
            )
        elif key in ["query_input_ids", "query_attn_mask"]:
            padded, _ = pad_sequences_1d([e["model_inputs"][key] for e in batch], dtype=torch.long)
            batched_data[key] = padded

    if "st_ed_indices" in model_inputs_keys:
        st_ed_indices = [e["model_inputs"]["st_ed_indices"] for e in batch]
        batched_data["st_ed_indices"] = torch.stack(st_ed_indices, dim=0)

        match_labels = np.zeros((len(st_ed_indices), fixed_length), dtype=np.int32)
        for idx, st_ed_index in enumerate(st_ed_indices):
            st_ed = st_ed_index.cpu().numpy()
            st, ed = st_ed[0], st_ed[1]
            match_labels[idx][st:(ed + 1)] = 1
        batched_data["match_labels"] = torch.tensor(match_labels, dtype=torch.long)

    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = {}
    for key, value in batched_model_inputs.items():
        if "feat" in key:
            model_inputs[key] = value[0].to(device, non_blocking=non_blocking)
            model_inputs[key.replace("feat", "mask")] = value[1].to(device, non_blocking=non_blocking)
        else:
            model_inputs[key] = value.to(device, non_blocking=non_blocking)
    return model_inputs


if __name__ == '__main__':
    options = BaseOptions().parse()
