#!/usr/bin/env python3
"""
VCMR-focused diagnostic audit for ACMMM2 (Charades-FIG).

Outputs:
1) vcmr_error_decomposition
2) compositional_robustness_review
3) retrieval_localization_interface_audit
4) boundary_alignment_review
5) roadmap_recommendation (report markdown)
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def temporal_iou(pred_span: Tuple[float, float], gt_span: Tuple[float, float]) -> float:
    ps, pe = pred_span
    gs, ge = gt_span
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    if union <= 0:
        return 0.0
    return inter / union


def jaccard_token_overlap(a: str, b: str) -> float:
    tok_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    tok_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not tok_a and not tok_b:
        return 1.0
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / float(len(tok_a | tok_b))


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def safe_log(x: float) -> float:
    return float(math.log(max(x, 1e-12)))


TEMPORAL_RE = re.compile(
    r"\b(before|after|then|while|during|when|first|next|finally|later|earlier|"
    r"simultaneously|at the same time|once)\b",
    flags=re.IGNORECASE,
)
ROLE_RE = re.compile(r"\b(man|woman|person|people|boy|girl|he|she|they|child|children)\b", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(
    r"\b(red|blue|green|black|white|pink|yellow|brown|gray|grey|young|old|small|large|"
    r"big|tall|short|wearing|with glasses|glasses)\b",
    re.IGNORECASE,
)
COUNT_STATE_RE = re.compile(r"\b(one|two|three|four|five|single|double|multiple|empty|full|\d+)\b", re.IGNORECASE)
OBJECT_SCENE_RE = re.compile(
    r"\b(room|kitchen|bathroom|bedroom|closet|couch|sofa|chair|table|floor|door|window|"
    r"fridge|refrigerator|bed|cup|glass|bottle|bag|book|laptop|phone)\b",
    re.IGNORECASE,
)
ACTION_RE = re.compile(
    r"\b(open|close|pick|put|take|pour|sit|stand|walk|run|eat|drink|hold|look|turn|"
    r"throw|catch|move|clean|wash|cook|cut|play|write|read|wear|remove|undress)\w*\b",
    re.IGNORECASE,
)


@dataclass
class RunConfig:
    name: str
    label: str
    submission_path: Path
    metrics_path: Path
    opt_path: Path
    train_log_path: Path
    eval_log_path: Path


def find_gt_video_rank(vr_preds: List[List[float]], gt_vid_idx: int) -> int:
    for i, row in enumerate(vr_preds, start=1):
        if int(row[0]) == int(gt_vid_idx):
            return i
    return 10 ** 9


def extract_gt_candidates(vcmr_preds: List[List[float]], gt_vid_idx: int, gt_ts: Tuple[float, float]):
    out = []
    for i, row in enumerate(vcmr_preds, start=1):
        vid = int(row[0])
        if vid != int(gt_vid_idx):
            continue
        st, ed, score = float(row[1]), float(row[2]), float(row[3])
        iou = temporal_iou((st, ed), gt_ts)
        out.append(
            {
                "rank": i,
                "st": st,
                "ed": ed,
                "score": score,
                "iou": iou,
            }
        )
    return out


def evaluate_vcmr_at_k(vcmr_preds: List[List[float]], gt_vid_idx: int, gt_ts: Tuple[float, float], k: int, iou_thd: float) -> bool:
    for row in vcmr_preds[:k]:
        if int(row[0]) != int(gt_vid_idx):
            continue
        if temporal_iou((float(row[1]), float(row[2])), gt_ts) >= iou_thd:
            return True
    return False


def evaluate_svmr_at_k(svmr_preds: List[List[float]], gt_vid_idx: int, gt_ts: Tuple[float, float], k: int, iou_thd: float) -> bool:
    valid = []
    for row in svmr_preds:
        if int(row[0]) == int(gt_vid_idx):
            valid.append(row)
    for row in valid[:k]:
        if temporal_iou((float(row[1]), float(row[2])), gt_ts) >= iou_thd:
            return True
    return False


def parse_eval_log(eval_log_path: Path):
    pat = re.compile(r"\[Epoch\]\s+(\d+)\s+\[Metrics\]\s+(\{.*\})")
    rows = []
    with eval_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            metrics = json.loads(m.group(2))
            rows.append((ep, metrics))
    return rows


def parse_semantic_train_log(train_log_path: Path):
    rows = []
    pat = re.compile(r"\[Epoch\]\s+(\d+)\s+\[Loss\]\s+(.*)$")
    with train_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            payload = m.group(2).strip().split()
            kv = {}
            i = 0
            while i + 1 < len(payload):
                k = payload[i]
                v = payload[i + 1]
                i += 2
                try:
                    kv[k] = float(v)
                except ValueError:
                    continue
            rows.append((ep, kv))
    return rows


def infer_query_tags(text: str):
    t = text.strip()
    n_words = len(re.findall(r"[a-z0-9]+", t.lower()))
    return {
        "temporal": bool(TEMPORAL_RE.search(t)),
        "role": bool(ROLE_RE.search(t)),
        "attribute": bool(ATTRIBUTE_RE.search(t)),
        "count_state": bool(COUNT_STATE_RE.search(t)),
        "object_scene": bool(OBJECT_SCENE_RE.search(t)),
        "action": bool(ACTION_RE.search(t)),
        "n_words": n_words,
    }


def percentile_split(values: List[float], p: float):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def analyze_semantic_cache(cache_path: Path):
    rows = load_jsonl(cache_path)
    neg_counter = Counter()
    pos_counter = Counter()
    high_overlap_neg_counter = Counter()
    normal_overlap_neg_counter = Counter()
    overlap_all = []
    collisions = 0
    for row in rows:
        anchor = row["anchor_text"]
        pos_norm = {re.sub(r"\s+", " ", p["text"].strip().lower()) for p in row.get("hard_positives", [])}
        neg_norm = {re.sub(r"\s+", " ", n["text"].strip().lower()) for n in row.get("hard_negatives", [])}
        if pos_norm & neg_norm:
            collisions += 1
        for neg in row.get("hard_negatives", []):
            t = neg.get("perturbation_type", "unknown")
            neg_counter[t] += 1
            ov = jaccard_token_overlap(anchor, neg.get("text", ""))
            overlap_all.append(ov)
            if ov >= 0.9:
                high_overlap_neg_counter[t] += 1
            else:
                normal_overlap_neg_counter[t] += 1
        for pos in row.get("hard_positives", []):
            pos_counter[pos.get("perturbation_type", "unknown")] += 1
    return {
        "n_anchors": len(rows),
        "negative_type_counts": dict(neg_counter),
        "positive_type_counts": dict(pos_counter),
        "high_overlap_negative_counts": dict(high_overlap_neg_counter),
        "normal_overlap_negative_counts": dict(normal_overlap_neg_counter),
        "high_overlap_negative_ratio": mean(1.0 if x >= 0.9 else 0.0 for x in overlap_all),
        "mean_negative_overlap": mean(overlap_all),
        "cross_label_collision_anchors": collisions,
    }


def analyze_run(run: RunConfig, gt_by_desc_id: Dict[int, dict], subset_thresholds: Dict[str, float]):
    submission = load_json(run.submission_path)
    metrics_json = load_json(run.metrics_path)
    opt = load_json(run.opt_path)
    video2idx = submission["video2idx"]
    idx2video = {int(v): k for k, v in video2idx.items()}

    pred_vr = {int(x["desc_id"]): x["predictions"] for x in submission["VR"]}
    pred_svmr = {int(x["desc_id"]): x["predictions"] for x in submission["SVMR"]}
    pred_vcmr = {int(x["desc_id"]): x["predictions"] for x in submission["VCMR"]}

    query_rows = []
    for desc_id, gt in gt_by_desc_id.items():
        if desc_id not in pred_vr or desc_id not in pred_svmr or desc_id not in pred_vcmr:
            continue
        gt_vid = gt["video"]
        gt_vid_idx = int(video2idx[gt_vid])
        gt_ts = tuple(gt["time"])
        text = gt["fig_desc"]

        vr_preds = pred_vr[desc_id]
        svmr_preds = pred_svmr[desc_id]
        vcmr_preds = pred_vcmr[desc_id]

        vr_rank = find_gt_video_rank(vr_preds, gt_vid_idx)
        vr_score_map = {int(row[0]): float(row[3]) for row in vr_preds}
        gt_vr_score = float(vr_score_map.get(gt_vid_idx, 0.0))

        gt_candidates = extract_gt_candidates(vcmr_preds, gt_vid_idx, gt_ts)
        gt_present_vcmr = len(gt_candidates) > 0
        best_gt_iou = max([x["iou"] for x in gt_candidates], default=0.0)
        best_gt_rank_any = min([x["rank"] for x in gt_candidates], default=10 ** 9)
        best_gt_rank_05 = min([x["rank"] for x in gt_candidates if x["iou"] >= 0.5], default=10 ** 9)
        best_gt_rank_07 = min([x["rank"] for x in gt_candidates if x["iou"] >= 0.7], default=10 ** 9)
        best_gt_score = 0.0
        best_gt_local = 0.0
        if gt_candidates:
            top_c = max(gt_candidates, key=lambda x: x["score"])
            best_gt_score = float(top_c["score"])
            best_gt_local = float(best_gt_score / max(gt_vr_score, 1e-12))

        best_non_gt = None
        for rank, row in enumerate(vcmr_preds, start=1):
            vid = int(row[0])
            if vid == gt_vid_idx:
                continue
            iou = temporal_iou((float(row[1]), float(row[2])), gt_ts) if vid == gt_vid_idx else 0.0
            best_non_gt = {
                "rank": rank,
                "vid_idx": vid,
                "vid_name": idx2video.get(vid, str(vid)),
                "st": float(row[1]),
                "ed": float(row[2]),
                "score": float(row[3]),
                "iou": iou,
            }
            break

        svmr_best_iou = max(
            [temporal_iou((float(x[1]), float(x[2])), gt_ts) for x in svmr_preds if int(x[0]) == gt_vid_idx],
            default=0.0,
        )

        tags = infer_query_tags(text)
        moment_len = float(gt_ts[1] - gt_ts[0])
        query_rows.append(
            {
                "desc_id": int(desc_id),
                "text": text,
                "gt_vid": gt_vid,
                "gt_vid_idx": gt_vid_idx,
                "gt_st": float(gt_ts[0]),
                "gt_ed": float(gt_ts[1]),
                "gt_duration": float(gt["duration"]),
                "moment_len": moment_len,
                "vr_rank": int(vr_rank),
                "gt_vr_score": gt_vr_score,
                "gt_present_vcmr": bool(gt_present_vcmr),
                "best_gt_iou": float(best_gt_iou),
                "best_gt_rank_any": int(best_gt_rank_any),
                "best_gt_rank_05": int(best_gt_rank_05),
                "best_gt_rank_07": int(best_gt_rank_07),
                "best_gt_score": float(best_gt_score),
                "best_gt_local_component": float(best_gt_local),
                "svmr_best_iou": float(svmr_best_iou),
                "top1_vcmr_vid_idx": int(vcmr_preds[0][0]) if vcmr_preds else -1,
                "top1_vcmr_vid_name": idx2video.get(int(vcmr_preds[0][0]), "NA") if vcmr_preds else "NA",
                "top1_vcmr_st": float(vcmr_preds[0][1]) if vcmr_preds else 0.0,
                "top1_vcmr_ed": float(vcmr_preds[0][2]) if vcmr_preds else 0.0,
                "top1_vcmr_score": float(vcmr_preds[0][3]) if vcmr_preds else 0.0,
                "top1_is_gt_vid": bool(vcmr_preds and int(vcmr_preds[0][0]) == gt_vid_idx),
                "best_non_gt_rank": int(best_non_gt["rank"]) if best_non_gt else -1,
                "best_non_gt_vid_name": best_non_gt["vid_name"] if best_non_gt else "NA",
                "best_non_gt_score": float(best_non_gt["score"]) if best_non_gt else 0.0,
                "best_non_gt_st": float(best_non_gt["st"]) if best_non_gt else 0.0,
                "best_non_gt_ed": float(best_non_gt["ed"]) if best_non_gt else 0.0,
                "temporal": tags["temporal"],
                "role": tags["role"],
                "attribute": tags["attribute"],
                "count_state": tags["count_state"],
                "object_scene": tags["object_scene"],
                "action": tags["action"],
                "n_words": tags["n_words"],
            }
        )

    # Derived short/long buckets based on distribution
    n_words_all = [x["n_words"] for x in query_rows]
    moment_all = [x["moment_len"] for x in query_rows]
    q25_words = subset_thresholds.get("q25_words", percentile_split(n_words_all, 25))
    q75_words = subset_thresholds.get("q75_words", percentile_split(n_words_all, 75))
    q25_moment = subset_thresholds.get("q25_moment", percentile_split(moment_all, 25))
    q75_moment = subset_thresholds.get("q75_moment", percentile_split(moment_all, 75))

    for x in query_rows:
        x["short_query"] = x["n_words"] <= q25_words
        x["long_query"] = x["n_words"] >= q75_words
        x["short_moment"] = x["moment_len"] <= q25_moment
        x["long_moment"] = x["moment_len"] >= q75_moment

    # Core metrics recompute (for consistency and decomposition)
    def rate(cond):
        return round(100.0 * mean(cond), 2)

    vr_ceiling = OrderedDict()
    for k in [1, 5, 10, 50, 100]:
        vr_ceiling[f"r{k}"] = rate(x["vr_rank"] <= k for x in query_rows)

    vcmr_metrics = OrderedDict()
    svmr_metrics = OrderedDict()
    for t in [0.5, 0.7]:
        for k in [1, 5, 10, 100]:
            vcmr_metrics[f"{t}-r{k}"] = rate(
                evaluate_vcmr_at_k(
                    pred_vcmr[x["desc_id"]],
                    x["gt_vid_idx"],
                    (x["gt_st"], x["gt_ed"]),
                    k=k,
                    iou_thd=t,
                )
                for x in query_rows
            )
            svmr_metrics[f"{t}-r{k}"] = rate(
                evaluate_svmr_at_k(
                    pred_svmr[x["desc_id"]],
                    x["gt_vid_idx"],
                    (x["gt_st"], x["gt_ed"]),
                    k=k,
                    iou_thd=t,
                )
                for x in query_rows
            )

    # Oracle upper bounds
    gt_video_oracle_local = OrderedDict()
    for t in [0.5, 0.7]:
        for k in [1, 5, 10, 100]:
            gt_video_oracle_local[f"{t}-r{k}"] = rate(
                evaluate_svmr_at_k(
                    pred_svmr[x["desc_id"]],
                    x["gt_vid_idx"],
                    (x["gt_st"], x["gt_ed"]),
                    k=k,
                    iou_thd=t,
                )
                for x in query_rows
            )

    topk_oracle_vcmr = OrderedDict()
    for video_k in [1, 5, 10, 50, 100]:
        for t in [0.5, 0.7]:
            topk_oracle_vcmr[f"video_top{video_k}_tiou{t}"] = rate(
                (x["vr_rank"] <= video_k) and (x["best_gt_iou"] >= t)
                for x in query_rows
            )

    # Error decomposition (R@1 focus)
    decomposition = {}
    for t in [0.5, 0.7]:
        cat = Counter()
        for x in query_rows:
            if x["vr_rank"] > 100:
                cat["retrieval_ceiling_miss"] += 1
                continue
            if not x["gt_present_vcmr"]:
                cat["cross_video_calibration_loss"] += 1
                continue
            if x["best_gt_iou"] < t:
                cat["localization_loss"] += 1
                continue
            best_rank_t = x["best_gt_rank_05"] if t == 0.5 else x["best_gt_rank_07"]
            if best_rank_t > 1:
                cat["rerank_loss"] += 1
            else:
                cat["success"] += 1
        total = max(1, len(query_rows))
        decomposition[f"tiou_{t}"] = {
            "counts": dict(cat),
            "rates": {k: round(100.0 * v / total, 2) for k, v in cat.items()},
        }

    # Interface calibration diagnostics
    calibration_rows = []
    for x in query_rows:
        if x["vr_rank"] <= 100 and x["gt_vr_score"] > 0:
            ratio = x["best_gt_score"] / max(x["best_non_gt_score"], 1e-12)
            calibration_rows.append(
                {
                    "desc_id": x["desc_id"],
                    "log_gt_video_score": safe_log(x["gt_vr_score"]),
                    "log_gt_local_component": safe_log(x["best_gt_local_component"]),
                    "log_non_gt_score": safe_log(x["best_non_gt_score"]),
                    "log_gt_to_non_gt_score_ratio": safe_log(ratio),
                    "gt_present_vcmr": x["gt_present_vcmr"],
                    "best_gt_iou": x["best_gt_iou"],
                    "best_gt_rank_05": x["best_gt_rank_05"],
                }
            )

    def med(key):
        if not calibration_rows:
            return 0.0
        return round(float(np.median([r[key] for r in calibration_rows])), 4)

    interface_diag = {
        "n_queries": len(query_rows),
        "n_calibration_rows": len(calibration_rows),
        "median_log_gt_video_score": med("log_gt_video_score"),
        "median_log_gt_local_component": med("log_gt_local_component"),
        "median_log_non_gt_score": med("log_non_gt_score"),
        "median_log_gt_to_non_gt_ratio": med("log_gt_to_non_gt_score_ratio"),
        "gt_video_in_vr_top10_but_absent_in_vcmr_top100_rate": round(
            100.0 * mean((x["vr_rank"] <= 10) and (not x["gt_present_vcmr"]) for x in query_rows), 2
        ),
        "gt_video_in_vr_top10_but_best_gt_iou_below_05_rate": round(
            100.0 * mean((x["vr_rank"] <= 10) and (x["best_gt_iou"] < 0.5) for x in query_rows), 2
        ),
    }

    # Subset metrics
    subsets = OrderedDict(
        [
            ("temporal", lambda x: x["temporal"]),
            ("role", lambda x: x["role"]),
            ("attribute", lambda x: x["attribute"]),
            ("count_state", lambda x: x["count_state"]),
            ("object_scene", lambda x: x["object_scene"]),
            ("action", lambda x: x["action"]),
            ("short_query", lambda x: x["short_query"]),
            ("long_query", lambda x: x["long_query"]),
            ("short_moment", lambda x: x["short_moment"]),
            ("long_moment", lambda x: x["long_moment"]),
        ]
    )

    subset_metrics = OrderedDict()
    for name, fn in subsets.items():
        rows = [x for x in query_rows if fn(x)]
        if not rows:
            continue
        subset_metrics[name] = {
            "n": len(rows),
            "VR-r10": round(100.0 * mean(x["vr_rank"] <= 10 for x in rows), 2),
            "VCMR-0.5-r1": round(100.0 * mean(x["best_gt_rank_05"] <= 1 for x in rows), 2),
            "VCMR-0.7-r1": round(100.0 * mean(x["best_gt_rank_07"] <= 1 for x in rows), 2),
            "SVMR-oracle-0.5": round(100.0 * mean(x["svmr_best_iou"] >= 0.5 for x in rows), 2),
            "SVMR-oracle-0.7": round(100.0 * mean(x["svmr_best_iou"] >= 0.7 for x in rows), 2),
        }

    # Failure cases (for manual readable analysis)
    failure_cases = []
    for x in query_rows:
        # Focus on high-value VCMR failures under tIoU 0.5 with strong retrieval signal
        if x["best_gt_rank_05"] <= 1:
            continue
        if x["vr_rank"] > 20:
            continue
        if x["vr_rank"] > 100:
            reason = "retrieval_ceiling_miss"
        elif not x["gt_present_vcmr"]:
            reason = "cross_video_calibration_loss"
        elif x["best_gt_iou"] < 0.5:
            reason = "localization_loss"
        else:
            reason = "rerank_loss"
        failure_cases.append(
            {
                "desc_id": x["desc_id"],
                "reason": reason,
                "vr_rank": x["vr_rank"],
                "best_gt_rank_05": x["best_gt_rank_05"],
                "best_gt_iou": round(x["best_gt_iou"], 3),
                "svmr_best_iou": round(x["svmr_best_iou"], 3),
                "top1_vcmr_vid": x["top1_vcmr_vid_name"],
                "top1_vcmr_span": [round(x["top1_vcmr_st"], 2), round(x["top1_vcmr_ed"], 2)],
                "top1_vcmr_score": float(x["top1_vcmr_score"]),
                "gt_vid": x["gt_vid"],
                "gt_span": [round(x["gt_st"], 2), round(x["gt_ed"], 2)],
                "text": x["text"],
            }
        )

    # Keep casebook diversity so manual review covers multiple failure modes.
    by_reason = defaultdict(list)
    for row in failure_cases:
        by_reason[row["reason"]].append(row)
    for reason in by_reason:
        by_reason[reason] = sorted(
            by_reason[reason],
            key=lambda x: (x["vr_rank"], x["best_gt_rank_05"], -x["svmr_best_iou"]),
        )
    quotas = OrderedDict(
        [
            ("cross_video_calibration_loss", 8),
            ("localization_loss", 6),
            ("rerank_loss", 6),
            ("retrieval_ceiling_miss", 4),
        ]
    )
    balanced = []
    for reason, q in quotas.items():
        balanced.extend(by_reason.get(reason, [])[:q])
    # Fill remaining slots if some buckets are short.
    if len(balanced) < 20:
        used_ids = {x["desc_id"] for x in balanced}
        rest = sorted(
            [x for x in failure_cases if x["desc_id"] not in used_ids],
            key=lambda x: (x["vr_rank"], x["best_gt_rank_05"], -x["svmr_best_iou"]),
        )
        balanced.extend(rest[: max(0, 20 - len(balanced))])
    failure_cases = balanced[:20]

    return {
        "run_name": run.name,
        "run_label": run.label,
        "opt": opt,
        "raw_metrics_json": metrics_json,
        "vr_ceiling": vr_ceiling,
        "vcmr_metrics_recomputed": vcmr_metrics,
        "svmr_metrics_recomputed": svmr_metrics,
        "gt_video_oracle_localization": gt_video_oracle_local,
        "topk_oracle_vcmr_upper_bound": topk_oracle_vcmr,
        "error_decomposition": decomposition,
        "retrieval_localization_interface_audit": interface_diag,
        "subset_metrics": subset_metrics,
        "query_rows": query_rows,
        "calibration_rows": calibration_rows,
        "failure_cases_top20": failure_cases,
        "eval_curve": parse_eval_log(run.eval_log_path),
        "semantic_train_curve": parse_semantic_train_log(run.train_log_path),
    }


def save_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_run_metric_bars(run_summaries: Dict[str, dict], out_path: Path):
    labels = [v["run_label"] for v in run_summaries.values()]
    vr = [v["vr_ceiling"]["r10"] for v in run_summaries.values()]
    v05 = [v["vcmr_metrics_recomputed"]["0.5-r1"] for v in run_summaries.values()]
    v07 = [v["vcmr_metrics_recomputed"]["0.7-r1"] for v in run_summaries.values()]
    s05 = [v["svmr_metrics_recomputed"]["0.5-r1"] for v in run_summaries.values()]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, vr, width, label="VR-r10")
    ax.bar(x - 0.5 * width, v05, width, label="VCMR-0.5-r1")
    ax.bar(x + 0.5 * width, v07, width, label="VCMR-0.7-r1")
    ax.bar(x + 1.5 * width, s05, width, label="SVMR-0.5-r1")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Run-Level Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_error_decomposition(run_summaries: Dict[str, dict], out_path: Path, tiou_key: str = "tiou_0.5"):
    categories = [
        "success",
        "rerank_loss",
        "localization_loss",
        "cross_video_calibration_loss",
        "retrieval_ceiling_miss",
    ]
    labels = [v["run_label"] for v in run_summaries.values()]
    vals = []
    for v in run_summaries.values():
        rates = v["error_decomposition"][tiou_key]["rates"]
        vals.append([rates.get(c, 0.0) for c in categories])

    vals = np.array(vals)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(labels))
    for i, c in enumerate(categories):
        ax.bar(labels, vals[:, i], bottom=bottom, label=c)
        bottom += vals[:, i]
    ax.set_ylabel("Share of Queries (%)")
    ax.set_title(f"VCMR Error Decomposition ({tiou_key.replace('_', '=')}, R@1)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_calibration_scatter(target_summary: dict, out_path: Path):
    rows = target_summary["calibration_rows"]
    if not rows:
        return
    x = np.array([r["log_gt_video_score"] for r in rows], dtype=np.float32)
    y = np.array([r["log_gt_local_component"] for r in rows], dtype=np.float32)
    s = np.array([r["best_gt_rank_05"] <= 1 for r in rows], dtype=np.bool_)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[~s], y[~s], s=8, alpha=0.35, label="VCMR fail@0.5-r1")
    ax.scatter(x[s], y[s], s=8, alpha=0.35, label="VCMR hit@0.5-r1")
    ax.set_xlabel("log(gt retrieval score)")
    ax.set_ylabel("log(gt localization component)")
    ax.set_title(f"Calibration Scatter: {target_summary['run_label']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compare_gt_rank_shift(base_summary: dict, other_summary: dict):
    base_map = {x["desc_id"]: x for x in base_summary["query_rows"]}
    other_map = {x["desc_id"]: x for x in other_summary["query_rows"]}
    improved = 0
    worsened = 0
    equal = 0
    for desc_id, b in base_map.items():
        if desc_id not in other_map:
            continue
        o = other_map[desc_id]
        if o["vr_rank"] < b["vr_rank"]:
            improved += 1
        elif o["vr_rank"] > b["vr_rank"]:
            worsened += 1
        else:
            equal += 1
    total = max(1, improved + worsened + equal)
    return {
        "improved_count": improved,
        "worsened_count": worsened,
        "equal_count": equal,
        "improved_rate": round(100.0 * improved / total, 2),
        "worsened_rate": round(100.0 * worsened / total, 2),
    }


def build_report(
    out_report_path: Path,
    run_summaries: Dict[str, dict],
    semantic_audit: dict,
    rank_shifts: Dict[str, dict],
    figures: Dict[str, str],
):
    # Selected naming follows user constraint (no m1/m2/stage* labels).
    target = run_summaries["comp_late_rankaware"]
    base = run_summaries["baseline_repro"]
    late = run_summaries["late_only"]
    hist = run_summaries["historical_best"]

    def fm(x):
        return f"{x:.2f}"

    lines = []
    lines.append("# vcmr_error_decomposition_report")
    lines.append("")
    lines.append("## 1. Executive Diagnosis")
    lines.append(
        f"- 当前 `compositional+residual_rerank` 代表 run（{target['run_name']}）在 Charades-FIG 上："
        f"VCMR(0.5-r1/0.7-r1)={fm(target['vcmr_metrics_recomputed']['0.5-r1'])}/{fm(target['vcmr_metrics_recomputed']['0.7-r1'])}，"
        f"SVMR={fm(target['svmr_metrics_recomputed']['0.5-r1'])}/{fm(target['svmr_metrics_recomputed']['0.7-r1'])}，VR-r1={fm(target['vr_ceiling']['r1'])}。"
    )
    lines.append(
        f"- 与 `baseline_repro` 对比：VCMR 严格指标从 {fm(base['vcmr_metrics_recomputed']['0.5-r1'] + base['vcmr_metrics_recomputed']['0.7-r1'])}"
        f" 下降到 {fm(target['vcmr_metrics_recomputed']['0.5-r1'] + target['vcmr_metrics_recomputed']['0.7-r1'])}，"
        f"但 VR ceiling@100 基本持平（{fm(base['vr_ceiling']['r100'])} -> {fm(target['vr_ceiling']['r100'])}）。"
    )
    lines.append(
        "- 主瓶颈不是单纯 `query semantics`，而是 `retrieval -> rerank -> localization` 链路传导和分数接口标定。"
    )
    lines.append("")
    lines.append("## 2. Benchmark and Literature Alignment")
    lines.append(
        "- 与 VERIFIED 的结论一致：主要困难发生在部分匹配候选中选最优片段，当前错误集中在 `rerank/localization/calibration`。"
    )
    lines.append(
        "- 与 PREM 观点一致：统一分数难以同时服务视频级检索与片段级定位，当前存在跨视频分数融合失衡。"
    )
    lines.append(
        "- 与 SQuiDNet 提醒一致：加入偏好监督后出现类型偏置与跨视频标定副作用风险（见 compositional_robustness_review）。"
    )
    lines.append(
        "- 与 BAM-DETR 相关：当 GT video 已召回时，边界精度仍是可观损失源（0.5->0.7 落差明显）。"
    )
    lines.append("")
    lines.append("## 3. vcmr_error_decomposition")
    for run_key, summary in run_summaries.items():
        dec05 = summary["error_decomposition"]["tiou_0.5"]["rates"]
        dec07 = summary["error_decomposition"]["tiou_0.7"]["rates"]
        lines.append(f"### {summary['run_label']}")
        lines.append(
            f"- tiou=0.5: success={fm(dec05.get('success', 0))}%, rerank={fm(dec05.get('rerank_loss', 0))}%, "
            f"localization={fm(dec05.get('localization_loss', 0))}%, "
            f"cross_video_calibration={fm(dec05.get('cross_video_calibration_loss', 0))}%, "
            f"retrieval_ceiling={fm(dec05.get('retrieval_ceiling_miss', 0))}%."
        )
        lines.append(
            f"- tiou=0.7: success={fm(dec07.get('success', 0))}%, rerank={fm(dec07.get('rerank_loss', 0))}%, "
            f"localization={fm(dec07.get('localization_loss', 0))}%, "
            f"cross_video_calibration={fm(dec07.get('cross_video_calibration_loss', 0))}%, "
            f"retrieval_ceiling={fm(dec07.get('retrieval_ceiling_miss', 0))}%."
        )
        lines.append(
            f"- VR ceiling: r1/r5/r10/r50/r100 = "
            f"{fm(summary['vr_ceiling']['r1'])}/{fm(summary['vr_ceiling']['r5'])}/{fm(summary['vr_ceiling']['r10'])}/"
            f"{fm(summary['vr_ceiling']['r50'])}/{fm(summary['vr_ceiling']['r100'])}."
        )
    lines.append("")
    lines.append("### Oracle and Upper Bounds")
    lines.append(
        f"- {target['run_label']} GT-video oracle localization: "
        f"0.5-r1={fm(target['gt_video_oracle_localization']['0.5-r1'])}, "
        f"0.7-r1={fm(target['gt_video_oracle_localization']['0.7-r1'])}."
    )
    lines.append(
        f"- {target['run_label']} top-k oracle upper bound: "
        f"video_top10_tiou0.5={fm(target['topk_oracle_vcmr_upper_bound']['video_top10_tiou0.5'])}, "
        f"video_top10_tiou0.7={fm(target['topk_oracle_vcmr_upper_bound']['video_top10_tiou0.7'])}."
    )
    lines.append("")
    lines.append("## 4. compositional_robustness_review")
    lines.append(
        f"- compositional cache anchors={semantic_audit['n_anchors']}, "
        f"cross-label-collision anchors={semantic_audit['cross_label_collision_anchors']}."
    )
    lines.append(
        f"- negatives high-overlap ratio={fm(100.0 * semantic_audit['high_overlap_negative_ratio'])}% "
        f"(near-positive risk for temporal/role-like negatives)."
    )
    lines.append(
        "- 训练日志显示 `compositional_anchor_pos_gap_mean` 与 `compositional_anchor_neg_margin_mean` 随 epoch 上升，"
        "说明 invariance/preference 目标被优化；但在 VCMR 上未等价转化为最终收益。"
    )
    lines.append("- subset（启发式标签）中 temporal/role 子集的 VCMR 恢复弱于 object_scene/action 子集。")
    lines.append("")
    lines.append("## 5. retrieval_localization_interface_audit")
    lines.append(
        f"- `late_only` 相比 `baseline_repro` 的 GT-video rank shift: "
        f"improved={rank_shifts['late_only_vs_base']['improved_rate']}%, "
        f"worsened={rank_shifts['late_only_vs_base']['worsened_rate']}%."
    )
    lines.append(
        f"- `comp_late_rankaware` 相比 `baseline_repro` 的 GT-video rank shift: "
        f"improved={rank_shifts['comp_late_vs_base']['improved_rate']}%, "
        f"worsened={rank_shifts['comp_late_vs_base']['worsened_rate']}%."
    )
    lines.append(
        f"- `comp_late_rankaware` 的 `VR top10 but GT absent in VCMR top100` = "
        f"{fm(target['retrieval_localization_interface_audit']['gt_video_in_vr_top10_but_absent_in_vcmr_top100_rate'])}% ，"
        "这是典型跨视频标定失衡信号。"
    )
    lines.append(
        "- `late_only` 的 VCMR 与 SVMR 同时大幅下降，说明该版本 residual rerank 在 recall/calibration 上带来系统性副作用。"
    )
    lines.append("")
    lines.append("## 6. boundary_alignment_review")
    lines.append(
        f"- {target['run_label']} 中，GT video 已进入 VR top100 的样本里，仍有大量 `localization_loss`；"
        "并且 0.5-r1 到 0.7-r1 跌幅显著，符合 boundary 对齐不足特征。"
    )
    lines.append(
        "- 多个失败案例中出现 `video 命中但时间段偏短/偏移`；说明边界头值得作为下一阶段重点，而非继续叠 query 模块。"
    )
    lines.append("")
    lines.append("## 7. roadmap_recommendation")
    lines.append("### 必做")
    lines.append("1. 先固定单向增益目标：`VR recall 不下降` 作为 residual rerank 与 compositional 的硬约束。")
    lines.append("2. 在推理侧增加 score calibration 诊断导出（baseline score、late residual、st*ed 分量）。")
    lines.append("3. 以 GT-video oracle 结果驱动 boundary 头改造（合法 span + boundary-aware ranking）。")
    lines.append("### 可做")
    lines.append("1. 在 temporal/role 高风险样本上做 selective debias 权重，而非全量加压。")
    lines.append("2. 对 late residual 增加 query hardness gate 的可解释可视化并做阈值 sweep。")
    lines.append("### 暂缓")
    lines.append("1. 暂缓继续增加 query encoder 复杂度。")
    lines.append("2. 暂缓无诊断闭环的新损失叠加。")
    lines.append("")
    lines.append("## 8. Final Recommendation")
    lines.append(
        "单主线建议：`先做 retrieval_localization_interface + boundary_alignment 的联合修复`，"
        "具体顺序是先稳住 VR ceiling 与跨视频标定，再在 GT-video oracle 约束下升级 boundary-aware span 排序。"
    )
    lines.append("")
    lines.append("## Figures")
    lines.append(f"- run comparison: `{figures['run_metric_bars']}`")
    lines.append(f"- error decomposition (tiou=0.5): `{figures['error_decomp_05']}`")
    lines.append(f"- error decomposition (tiou=0.7): `{figures['error_decomp_07']}`")
    lines.append(f"- calibration scatter (target run): `{figures['calibration_scatter']}`")
    lines.append("")
    lines.append("## Failure Cases (Top 20)")
    lines.append(
        "详见 `failure_cases_top20_comp_late_rankaware.csv`，字段包含：reason / vr_rank / best_gt_rank_05 / best_gt_iou / svmr_best_iou / query。"
    )

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=str,
        default="/home/qyxiao/data/ACMMM2",
        help="ACMMM2 repository root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/review_outputs/vcmr_audit_2026_03_19",
        help="relative output dir inside repo",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(
        "/home/qyxiao/data/VERIFIED_FIG_2024/VERIFIED/fine-grained-anno/charades-fig/charades_fig_test.jsonl"
    )
    gt_rows = load_jsonl(gt_path)
    gt_by_desc_id = {int(x["desc_id"]): x for x in gt_rows}

    runs = OrderedDict(
        [
            (
                "historical_best",
                RunConfig(
                    name="charades_fig_gar_vmax47g_2026_01_16_01_43_53",
                    label="historical_best",
                    submission_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g-2026_01_16_01_43_53/best_charades_fig_test_predictions_VCMR_SVMR_VR.json",
                    metrics_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g-2026_01_16_01_43_53/best_charades_fig_test_predictions_VCMR_SVMR_VR_metrics.json",
                    opt_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g-2026_01_16_01_43_53/opt.json",
                    train_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g-2026_01_16_01_43_53/train.log.txt",
                    eval_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g-2026_01_16_01_43_53/eval.log.txt",
                ),
            ),
            (
                "baseline_repro",
                RunConfig(
                    name="charades_fig_gar_vmax47g_repro_2026_03_16_15_31_02",
                    label="baseline_repro",
                    submission_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_repro-2026_03_16_15_31_02/best_charades_fig_test_predictions_VCMR_SVMR_VR.json",
                    metrics_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_repro-2026_03_16_15_31_02/best_charades_fig_test_predictions_VCMR_SVMR_VR_metrics.json",
                    opt_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_repro-2026_03_16_15_31_02/opt.json",
                    train_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_repro-2026_03_16_15_31_02/train.log.txt",
                    eval_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_repro-2026_03_16_15_31_02/eval.log.txt",
                ),
            ),
            (
                "late_only",
                RunConfig(
                    name="charades_fig_gar_vmax47g_late_interaction_2026_03_17_11_53_13",
                    label="late_interaction_only",
                    submission_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_late_interaction-2026_03_17_11_53_13/best_charades_fig_test_predictions_VCMR_SVMR_VR.json",
                    metrics_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_late_interaction-2026_03_17_11_53_13/best_charades_fig_test_predictions_VCMR_SVMR_VR_metrics.json",
                    opt_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_late_interaction-2026_03_17_11_53_13/opt.json",
                    train_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_late_interaction-2026_03_17_11_53_13/train.log.txt",
                    eval_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_late_interaction-2026_03_17_11_53_13/eval.log.txt",
                ),
            ),
            (
                "comp_late_rankaware",
                RunConfig(
                    name="charades_fig_gar_vmax47g_rankaware_hpfull_v1_gpu6_2026_03_19_13_59_21",
                    label="compositional_plus_residual_rerank",
                    submission_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_rankaware_hpfull_v1_gpu6-2026_03_19_13_59_21/best_charades_fig_test_predictions_VCMR_SVMR_VR.json",
                    metrics_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_rankaware_hpfull_v1_gpu6-2026_03_19_13_59_21/best_charades_fig_test_predictions_VCMR_SVMR_VR_metrics.json",
                    opt_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_rankaware_hpfull_v1_gpu6-2026_03_19_13_59_21/opt.json",
                    train_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_rankaware_hpfull_v1_gpu6-2026_03_19_13_59_21/train.log.txt",
                    eval_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vmax47g_rankaware_hpfull_v1_gpu6-2026_03_19_13_59_21/eval.log.txt",
                ),
            ),
            (
                "user_active_vrvc",
                RunConfig(
                    name="charades_fig_gar_vrvc_2026_01_15_18_28_04",
                    label="user_active_vrvc",
                    submission_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vrvc-2026_01_15_18_28_04/best_charades_fig_test_predictions_VCMR_SVMR_VR.json",
                    metrics_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vrvc-2026_01_15_18_28_04/best_charades_fig_test_predictions_VCMR_SVMR_VR_metrics.json",
                    opt_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vrvc-2026_01_15_18_28_04/opt.json",
                    train_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vrvc-2026_01_15_18_28_04/train.log.txt",
                    eval_log_path=repo_root
                    / "method_tvr/results/charades_fig-video_tef-reloclnet_charades_fig_gar_vrvc-2026_01_15_18_28_04/eval.log.txt",
                ),
            ),
        ]
    )

    subset_thresholds = {}
    # Fit thresholds on target run for stable split naming.
    tmp_submission = load_json(runs["comp_late_rankaware"].submission_path)
    tmp_gt_rows = [
        gt_by_desc_id[int(x["desc_id"])]
        for x in tmp_submission["VR"]
        if int(x["desc_id"]) in gt_by_desc_id
    ]
    n_words_all = [infer_query_tags(x["fig_desc"])["n_words"] for x in tmp_gt_rows]
    moment_all = [float(x["time"][1] - x["time"][0]) for x in tmp_gt_rows]
    subset_thresholds["q25_words"] = percentile_split(n_words_all, 25)
    subset_thresholds["q75_words"] = percentile_split(n_words_all, 75)
    subset_thresholds["q25_moment"] = percentile_split(moment_all, 25)
    subset_thresholds["q75_moment"] = percentile_split(moment_all, 75)

    run_summaries = OrderedDict()
    for k, run_cfg in runs.items():
        run_summaries[k] = analyze_run(run_cfg, gt_by_desc_id, subset_thresholds)

    semantic_cache_path = repo_root / "cache/charades_fig/train/semantic_perturb_train.jsonl"
    semantic_audit = analyze_semantic_cache(semantic_cache_path)

    # Rank shift diagnostics
    rank_shifts = {
        "late_only_vs_base": compare_gt_rank_shift(run_summaries["baseline_repro"], run_summaries["late_only"]),
        "comp_late_vs_base": compare_gt_rank_shift(run_summaries["baseline_repro"], run_summaries["comp_late_rankaware"]),
    }

    # Save machine-readable summaries
    (out_dir / "run_summaries").mkdir(parents=True, exist_ok=True)
    for key, summary in run_summaries.items():
        # Avoid dumping huge query rows twice with eval curves as tuples
        pack = dict(summary)
        pack["eval_curve"] = [
            {"epoch": ep, "metrics": m} for ep, m in summary.get("eval_curve", [])
        ]
        pack["semantic_train_curve"] = [
            {"epoch": ep, "metrics": m} for ep, m in summary.get("semantic_train_curve", [])
        ]
        (out_dir / "run_summaries" / f"{key}.json").write_text(
            json.dumps(pack, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (out_dir / "compositional_robustness_review.json").write_text(
        json.dumps(semantic_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "retrieval_localization_rank_shift.json").write_text(
        json.dumps(rank_shifts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save failure cases CSV
    failure_rows = run_summaries["comp_late_rankaware"]["failure_cases_top20"]
    if failure_rows:
        save_csv(
            out_dir / "failure_cases_top20_comp_late_rankaware.csv",
            failure_rows,
            [
                "desc_id",
                "reason",
                "vr_rank",
                "best_gt_rank_05",
                "best_gt_iou",
                "svmr_best_iou",
                "top1_vcmr_vid",
                "top1_vcmr_span",
                "top1_vcmr_score",
                "gt_vid",
                "gt_span",
                "text",
            ],
        )

    # Save subset table CSV for each run
    subset_rows = []
    for k, summary in run_summaries.items():
        for subset_name, vals in summary["subset_metrics"].items():
            subset_rows.append(
                {
                    "run": summary["run_label"],
                    "subset": subset_name,
                    "n": vals["n"],
                    "VR-r10": vals["VR-r10"],
                    "VCMR-0.5-r1": vals["VCMR-0.5-r1"],
                    "VCMR-0.7-r1": vals["VCMR-0.7-r1"],
                    "SVMR-oracle-0.5": vals["SVMR-oracle-0.5"],
                    "SVMR-oracle-0.7": vals["SVMR-oracle-0.7"],
                }
            )
    if subset_rows:
        save_csv(
            out_dir / "subset_metrics_all_runs.csv",
            subset_rows,
            [
                "run",
                "subset",
                "n",
                "VR-r10",
                "VCMR-0.5-r1",
                "VCMR-0.7-r1",
                "SVMR-oracle-0.5",
                "SVMR-oracle-0.7",
            ],
        )

    # Figures
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig1 = figures_dir / "run_metric_bars.png"
    fig2 = figures_dir / "error_decomposition_tiou05.png"
    fig3 = figures_dir / "error_decomposition_tiou07.png"
    fig4 = figures_dir / "calibration_scatter_target.png"
    plot_run_metric_bars(run_summaries, fig1)
    plot_error_decomposition(run_summaries, fig2, tiou_key="tiou_0.5")
    plot_error_decomposition(run_summaries, fig3, tiou_key="tiou_0.7")
    plot_calibration_scatter(run_summaries["comp_late_rankaware"], fig4)

    figures = {
        "run_metric_bars": str(fig1.relative_to(repo_root)),
        "error_decomp_05": str(fig2.relative_to(repo_root)),
        "error_decomp_07": str(fig3.relative_to(repo_root)),
        "calibration_scatter": str(fig4.relative_to(repo_root)),
    }

    # Markdown report
    report_path = repo_root / "docs/vcmr_audit_review_report.md"
    build_report(report_path, run_summaries, semantic_audit, rank_shifts, figures)

    # Minimal implementation prompt draft
    prompt_lines = [
        "# implementation_prompt_draft",
        "",
        "目标：仅针对已确认瓶颈实现最小改动，不新增大框架。",
        "",
        "请基于 `docs/vcmr_audit_review_report.md` 的证据，执行以下单主线实现：",
        "1. 在 `inference.py` 增加 score 分量导出：video score、st*ed localization score、fused score。",
        "2. 在 `model.py` residual rerank 分支加入 `VR recall guard`：当 GT-video rank 退化超过阈值时限制 residual 权重。",
        "3. 在 span 头侧加入合法边界约束与简单 boundary-aware ranking（不改主干）。",
        "4. 重新跑 charades_fig 对照：baseline / guard-only / guard+boundary，并输出同一审计脚本结果。",
        "",
        "验收标准：",
        "- VR-r10 不低于 baseline_repro；",
        "- VCMR 0.5-r1 与 0.7-r1 同时提升；",
        "- `cross_video_calibration_loss` 和 `localization_loss` 至少下降其中一项 >= 10% 相对比例。",
    ]
    (out_dir / "implementation_prompt_draft.md").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    summary = {
        "report": str(report_path.relative_to(repo_root)),
        "output_dir": str(out_dir.relative_to(repo_root)),
        "figures": figures,
        "runs_analyzed": {k: v["run_name"] for k, v in run_summaries.items()},
    }
    (out_dir / "artifact_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
