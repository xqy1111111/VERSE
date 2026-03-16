import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from method_tvr.semantic_perturb.hashing import sha256_file
from method_tvr.semantic_perturb.losses import compute_consistency_loss, compute_preference_loss, severity_to_weight
from method_tvr.semantic_perturb.schema import (
    RELATION_HARD_NEGATIVE,
    RELATION_HARD_POSITIVE,
    is_verifier_accept,
    validate_verifier_response,
)
from utils.basic_utils import load_json, load_jsonl


class SemanticCacheLookup:
    def __init__(self, entries: Dict[int, Dict], manifest: Dict, strict_mode: bool, fail_on_missing: bool):
        self.entries = entries
        self.manifest = manifest
        self.strict_mode = bool(strict_mode)
        self.fail_on_missing = bool(fail_on_missing)

    def get_entry(self, desc_id: int) -> Optional[Dict]:
        key = int(desc_id)
        entry = self.entries.get(key)
        if entry is None and (self.strict_mode or self.fail_on_missing):
            raise RuntimeError("Semantic cache missing desc_id={} in strict mode".format(desc_id))
        return entry


class SemanticLossRuntime:
    def __init__(self, opt):
        self.enabled = bool(getattr(opt, "semantic_enable", False))
        self.strict_mode = bool(getattr(opt, "semantic_strict_mode", True))
        self.no_fallback = bool(getattr(opt, "semantic_no_fallback", True))
        self.num_hard_neg = int(getattr(opt, "semantic_num_hard_neg", 0))
        self.num_hard_pos = int(getattr(opt, "semantic_num_hard_pos", 0))
        self.use_preference_loss = bool(getattr(opt, "semantic_use_preference_loss", False))
        self.preference_margin = float(getattr(opt, "semantic_preference_margin", 0.2))
        self.preference_weight = float(getattr(opt, "semantic_preference_weight", 1.0))
        self.use_consistency_loss = bool(getattr(opt, "semantic_use_consistency_loss", False))
        self.consistency_weight = float(getattr(opt, "semantic_consistency_weight", 1.0))
        self.max_desc_l = int(getattr(opt, "max_desc_l", 30))
        self.normalize_tfeat = not bool(getattr(opt, "no_norm_tfeat", False))
        self.expected_hidden_size = int(getattr(opt, "q_feat_size", 768))

        self.tokenizer = None
        self.text_encoder = None

        if self.enabled:
            if not (self.use_preference_loss or self.use_consistency_loss):
                raise ValueError("semantic_enable=true requires at least one Semantic loss to be enabled")
            from transformers import AutoModel, AutoTokenizer

            text_encoder_name = getattr(opt, "semantic_text_encoder_name_or_path", None) or getattr(
                opt, "tokenizer_name_or_path", "bert-base-uncased"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, use_fast=True)
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            hidden_size = int(getattr(self.text_encoder.config, "hidden_size", -1))
            if hidden_size != self.expected_hidden_size:
                raise ValueError(
                    "Semantic text encoder hidden_size {} does not match q_feat_size {}".format(
                        hidden_size, self.expected_hidden_size
                    )
                )

    def _encode_texts(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            empty_feat = torch.zeros(0, self.max_desc_l, self.expected_hidden_size, device=device)
            empty_mask = torch.zeros(0, self.max_desc_l, device=device)
            return empty_feat, empty_mask

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_desc_l,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**encoded)
            hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].float()
        hidden = hidden[:, : self.max_desc_l]
        mask = mask[:, : self.max_desc_l]
        if hidden.size(1) < self.max_desc_l:
            pad_len = self.max_desc_l - hidden.size(1)
            hidden = F.pad(hidden, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))

        hidden = hidden.to(dtype=torch.float32)
        if self.normalize_tfeat:
            hidden = F.normalize(hidden, dim=-1)
        return hidden, mask

    @staticmethod
    def _validate_variant_item(item: Dict) -> None:
        required = ["text", "relation_label", "perturbation_type", "severity", "short_rationale", "verifier"]
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError("Semantic variant missing keys {}".format(missing))
        if item["relation_label"] not in {RELATION_HARD_NEGATIVE, RELATION_HARD_POSITIVE}:
            raise ValueError("Unsupported relation_label '{}'".format(item["relation_label"]))
        if int(item["severity"]) not in {1, 2, 3}:
            raise ValueError("Unsupported severity '{}'".format(item["severity"]))
        verifier_result = validate_verifier_response(item["verifier"])
        if not is_verifier_accept(verifier_result, item["relation_label"]):
            raise ValueError("Verifier result is not valid for relation_label '{}'".format(item["relation_label"]))

    def compute_losses(self, model_core, model_inputs: Dict, batch_meta: List[Dict], model_aux: Dict):
        if not self.enabled:
            zero = model_inputs["query_feat"].new_tensor(0.0)
            return zero, {
                "loss_semantic_pref": 0.0,
                "loss_semantic_cons": 0.0,
                "loss_semantic_total": 0.0,
            }

        query_context_scores = model_aux.get("query_context_scores")
        encoded_video_feat = model_aux.get("encoded_video_feat")
        if query_context_scores is None or encoded_video_feat is None:
            raise RuntimeError("Semantic loss requires query_context_scores and encoded_video_feat from model_aux")

        anchor_scores = torch.diagonal(query_context_scores, offset=0)
        video_mask = model_inputs["video_mask"]

        all_texts: List[str] = []
        plan = []

        for meta in batch_meta:
            semantic_entry = meta.get("semantic")
            if semantic_entry is None:
                if self.strict_mode or self.no_fallback:
                    raise RuntimeError("Semantic enabled but batch sample has no semantic entry")
                plan.append({"neg_idx": [], "pos_idx": [], "neg_sev": []})
                continue

            neg_items = list(semantic_entry.get("hard_negatives", []))
            pos_items = list(semantic_entry.get("hard_positives", []))

            if len(neg_items) < self.num_hard_neg and (self.strict_mode or self.no_fallback):
                raise RuntimeError("Semantic cache has insufficient hard_negatives for desc_id={}".format(meta.get("desc_id")))
            if len(pos_items) < self.num_hard_pos and (self.strict_mode or self.no_fallback):
                raise RuntimeError("Semantic cache has insufficient hard_positives for desc_id={}".format(meta.get("desc_id")))

            neg_items = neg_items[: self.num_hard_neg]
            pos_items = pos_items[: self.num_hard_pos]

            neg_idx = []
            pos_idx = []
            neg_sev = []

            for item in neg_items:
                self._validate_variant_item(item)
                neg_idx.append(len(all_texts))
                neg_sev.append(int(item["severity"]))
                all_texts.append(item["text"])

            for item in pos_items:
                self._validate_variant_item(item)
                pos_idx.append(len(all_texts))
                all_texts.append(item["text"])

            plan.append({"neg_idx": neg_idx, "pos_idx": pos_idx, "neg_sev": neg_sev})

        if not all_texts and (self.strict_mode or self.no_fallback):
            raise RuntimeError("Semantic enabled but no perturbation texts found in batch")

        if self.text_encoder is not None:
            encoder_device = next(self.text_encoder.parameters()).device
            if encoder_device != anchor_scores.device:
                self.text_encoder.to(anchor_scores.device)

        variant_feat, variant_mask = self._encode_texts(all_texts, device=anchor_scores.device)

        pos_scores_per_sample: List[torch.Tensor] = []
        neg_scores_per_sample: List[torch.Tensor] = []
        neg_weights_per_sample: List[torch.Tensor] = []

        for idx, sample in enumerate(plan):
            neg_idx = sample["neg_idx"]
            pos_idx = sample["pos_idx"]
            neg_sev = sample["neg_sev"]

            context_feat = encoded_video_feat[idx : idx + 1]
            context_mask = video_mask[idx : idx + 1]

            if neg_idx:
                neg_scores = model_core.score_queries_to_single_context(
                    query_feat=variant_feat[neg_idx],
                    query_mask=variant_mask[neg_idx],
                    context_feat=context_feat,
                    context_mask=context_mask,
                )
                neg_weights = neg_scores.new_tensor([severity_to_weight(x) for x in neg_sev])
            else:
                neg_scores = anchor_scores.new_zeros(0)
                neg_weights = anchor_scores.new_zeros(0)

            if pos_idx:
                pos_scores = model_core.score_queries_to_single_context(
                    query_feat=variant_feat[pos_idx],
                    query_mask=variant_mask[pos_idx],
                    context_feat=context_feat,
                    context_mask=context_mask,
                )
            else:
                pos_scores = anchor_scores.new_zeros(0)

            neg_scores_per_sample.append(neg_scores)
            neg_weights_per_sample.append(neg_weights)
            pos_scores_per_sample.append(pos_scores)

        preference_raw = anchor_scores.new_tensor(0.0)
        consistency_raw = anchor_scores.new_tensor(0.0)

        if self.use_preference_loss:
            preference_raw = compute_preference_loss(
                anchor_scores=anchor_scores,
                pos_scores_per_sample=pos_scores_per_sample,
                neg_scores_per_sample=neg_scores_per_sample,
                neg_weights_per_sample=neg_weights_per_sample,
                margin=self.preference_margin,
            )

        if self.use_consistency_loss:
            consistency_raw = compute_consistency_loss(anchor_scores=anchor_scores, pos_scores_per_sample=pos_scores_per_sample)

        preference_loss = self.preference_weight * preference_raw
        consistency_loss = self.consistency_weight * consistency_raw
        total = preference_loss + consistency_loss

        return total, {
            "loss_semantic_pref": float(preference_loss.detach().cpu().item()),
            "loss_semantic_cons": float(consistency_loss.detach().cpu().item()),
            "loss_semantic_total": float(total.detach().cpu().item()),
        }


def _infer_manifest_path(cache_path: str) -> str:
    if cache_path.endswith(".jsonl"):
        return cache_path[:-6] + ".manifest.json"
    return cache_path + ".manifest.json"


def _validate_manifest_against_training_opt(manifest: Dict, opt, source_data_path: str) -> None:
    if manifest.get("dataset") != getattr(opt, "dset_name", None):
        raise RuntimeError("Semantic manifest dataset mismatch: {} vs {}".format(manifest.get("dataset"), getattr(opt, "dset_name", None)))

    expected_split = getattr(opt, "semantic_cache_split", "train")
    if manifest.get("split") != expected_split:
        raise RuntimeError("Semantic manifest split mismatch: {} vs {}".format(manifest.get("split"), expected_split))

    source_hash = sha256_file(source_data_path)
    if manifest.get("source_hash") != source_hash:
        raise RuntimeError("Semantic manifest source_hash mismatch")

    checks = [
        ("prompt_version", getattr(opt, "semantic_prompt_version", "")),
        ("schema_version", getattr(opt, "semantic_schema_version", "")),
        ("generator_model", getattr(opt, "semantic_generator_model", "")),
        ("verifier_model", getattr(opt, "semantic_verifier_model", "")),
    ]
    for key, expected in checks:
        if manifest.get(key) != expected:
            raise RuntimeError("Semantic manifest {} mismatch: {} vs {}".format(key, manifest.get(key), expected))

    expected_neg_types = sorted(getattr(opt, "semantic_neg_types", []))
    expected_pos_types = sorted(getattr(opt, "semantic_pos_types", []))
    expected_severity_levels = sorted(int(x) for x in getattr(opt, "semantic_severity_levels", []))

    if sorted(manifest.get("neg_types", [])) != expected_neg_types:
        raise RuntimeError("Semantic manifest neg_types mismatch")
    if sorted(manifest.get("pos_types", [])) != expected_pos_types:
        raise RuntimeError("Semantic manifest pos_types mismatch")
    if sorted(int(x) for x in manifest.get("severity_levels", [])) != expected_severity_levels:
        raise RuntimeError("Semantic manifest severity_levels mismatch")

    if manifest.get("backend") != "llm":
        raise RuntimeError("Semantic manifest backend must be 'llm'")

    if bool(getattr(opt, "semantic_no_fallback", True)) and not bool(manifest.get("no_fallback", False)):
        raise RuntimeError("Semantic manifest no_fallback=false conflicts with training strict mode")


def _validate_cache_entry_shape(entry: Dict, num_hard_neg: int, num_hard_pos: int) -> None:
    required = ["desc_id", "anchor_text", "source_meta", "hard_negatives", "hard_positives", "build_meta"]
    missing = [k for k in required if k not in entry]
    if missing:
        raise ValueError("Semantic cache entry missing keys {}".format(missing))

    if not isinstance(entry["hard_negatives"], list) or not isinstance(entry["hard_positives"], list):
        raise ValueError("hard_negatives/hard_positives must be lists")

    if len(entry["hard_negatives"]) < num_hard_neg:
        raise ValueError("insufficient hard_negatives")
    if len(entry["hard_positives"]) < num_hard_pos:
        raise ValueError("insufficient hard_positives")

    for item in entry["hard_negatives"][:num_hard_neg]:
        SemanticLossRuntime._validate_variant_item(item)
        if item["relation_label"] != RELATION_HARD_NEGATIVE:
            raise ValueError("hard_negatives relation_label must be hard_negative")

    for item in entry["hard_positives"][:num_hard_pos]:
        SemanticLossRuntime._validate_variant_item(item)
        if item["relation_label"] != RELATION_HARD_POSITIVE:
            raise ValueError("hard_positives relation_label must be hard_positive")


def load_semantic_cache_lookup(opt, source_data_path: str) -> Optional[SemanticCacheLookup]:
    if not bool(getattr(opt, "semantic_enable", False)):
        return None

    if getattr(opt, "semantic_backend", "none") != "llm":
        raise ValueError("semantic_enable=true requires semantic_backend=llm")

    cache_path = str(getattr(opt, "semantic_cache_path", "") or "").strip()
    if not cache_path:
        raise RuntimeError("semantic_enable=true requires --semantic_cache_path")

    if not os.path.isfile(cache_path):
        if bool(getattr(opt, "semantic_fail_on_missing_cache", True)) or bool(getattr(opt, "semantic_no_fallback", True)):
            raise FileNotFoundError("Semantic cache file not found: {}".format(cache_path))
        return None

    manifest_path = _infer_manifest_path(cache_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("Semantic manifest file not found: {}".format(manifest_path))

    manifest = load_json(manifest_path)
    _validate_manifest_against_training_opt(manifest, opt, source_data_path)

    strict_invalid = (
        bool(getattr(opt, "semantic_fail_on_invalid_cache", True))
        or bool(getattr(opt, "semantic_strict_mode", True))
        or bool(getattr(opt, "semantic_no_fallback", True))
    )

    num_hard_neg = int(getattr(opt, "semantic_num_hard_neg", 0))
    num_hard_pos = int(getattr(opt, "semantic_num_hard_pos", 0))

    entries: Dict[int, Dict] = {}
    for raw in load_jsonl(cache_path):
        try:
            _validate_cache_entry_shape(raw, num_hard_neg=num_hard_neg, num_hard_pos=num_hard_pos)
        except Exception as err:  # noqa: PERF203
            if strict_invalid:
                raise RuntimeError("Invalid Semantic cache entry: {}".format(err)) from err
            continue
        entries[int(raw["desc_id"])] = raw

    return SemanticCacheLookup(
        entries=entries,
        manifest=manifest,
        strict_mode=bool(getattr(opt, "semantic_strict_mode", True)),
        fail_on_missing=bool(getattr(opt, "semantic_fail_on_missing_cache", True)),
    )
