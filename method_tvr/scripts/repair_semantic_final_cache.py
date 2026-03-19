#!/usr/bin/env python3
import argparse
import copy
import itertools
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from method_tvr.semantic_perturb.rewrite_sampler import normalize_rewrite_text, sanitize_and_sample_rewrites
from method_tvr.semantic_perturb.schema import (
    RELATION_HARD_NEGATIVE,
    RELATION_HARD_POSITIVE,
    is_verifier_accept,
    validate_verifier_response,
)


@dataclass(frozen=True)
class SamplerCfg:
    num_hard_neg: int
    num_hard_pos: int
    collision_sanitization_enabled: bool
    rewrite_type_quota_enabled: bool
    risky_negative_filter_enabled: bool
    risky_negative_overlap_threshold: float
    risky_negative_start_epoch: int
    risky_negative_downweight: float
    current_epoch: int


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_conf(item: Dict) -> float:
    try:
        return float(item.get("verifier", {}).get("confidence", 0.0))
    except Exception:
        return 0.0


def _is_variant_valid(item: Dict, expected_label: str) -> bool:
    if not isinstance(item, dict):
        return False
    required = ["text", "relation_label", "perturbation_type", "severity", "short_rationale", "verifier"]
    for key in required:
        if key not in item:
            return False
    if item.get("relation_label") != expected_label:
        return False
    if not normalize_rewrite_text(item.get("text", "")):
        return False
    try:
        severity = int(item.get("severity", -1))
    except Exception:
        return False
    if severity not in {1, 2, 3}:
        return False
    try:
        verifier_result = validate_verifier_response(item.get("verifier", {}))
    except Exception:
        return False
    return bool(is_verifier_accept(verifier_result, expected_label))


def _entry_load_valid(row: Dict, cfg: SamplerCfg) -> Tuple[bool, str]:
    neg = row.get("hard_negatives")
    pos = row.get("hard_positives")
    if not isinstance(neg, list) or not isinstance(pos, list):
        return False, "hard_negatives_or_hard_positives_not_list"
    if len(neg) < cfg.num_hard_neg:
        return False, "insufficient_hard_negatives_raw"
    if len(pos) < cfg.num_hard_pos:
        return False, "insufficient_hard_positives_raw"
    for i, item in enumerate(neg[: cfg.num_hard_neg]):
        if not _is_variant_valid(item, RELATION_HARD_NEGATIVE):
            return False, "invalid_hard_negative_at_{}".format(i)
    for i, item in enumerate(pos[: cfg.num_hard_pos]):
        if not _is_variant_valid(item, RELATION_HARD_POSITIVE):
            return False, "invalid_hard_positive_at_{}".format(i)
    return True, ""


def _dedupe_by_text(items: Iterable[Dict], expected_label: str) -> List[Dict]:
    # Keep the highest-confidence item for the same normalized text.
    best: Dict[str, Dict] = {}
    for item in items:
        if not _is_variant_valid(item, expected_label):
            continue
        key = normalize_rewrite_text(item.get("text", ""))
        if not key:
            continue
        cur = best.get(key)
        if cur is None or _safe_conf(item) > _safe_conf(cur):
            best[key] = copy.deepcopy(item)
    out = list(best.values())
    out.sort(
        key=lambda x: (
            -_safe_conf(x),
            int(x.get("severity", 0)),
            str(x.get("perturbation_type", "")),
            normalize_rewrite_text(x.get("text", "")),
        )
    )
    return out


def _evaluate(anchor_text: str, pos_items: Sequence[Dict], neg_items: Sequence[Dict], cfg: SamplerCfg) -> Dict:
    selected_pos, selected_neg, _, _, stats = sanitize_and_sample_rewrites(
        anchor_text=anchor_text,
        positive_rewrites=list(pos_items),
        negative_rewrites=list(neg_items),
        positive_sample_size=cfg.num_hard_pos,
        negative_sample_size=cfg.num_hard_neg,
        collision_sanitization_enabled=cfg.collision_sanitization_enabled,
        rewrite_type_quota_enabled=cfg.rewrite_type_quota_enabled,
        risky_negative_filter_enabled=cfg.risky_negative_filter_enabled,
        risky_negative_overlap_threshold=cfg.risky_negative_overlap_threshold,
        risky_negative_start_epoch=cfg.risky_negative_start_epoch,
        risky_negative_downweight=cfg.risky_negative_downweight,
        current_epoch=cfg.current_epoch,
    )
    selected_pos_valid = sum(1 for x in selected_pos if _is_variant_valid(x, RELATION_HARD_POSITIVE))
    selected_neg_valid = sum(1 for x in selected_neg if _is_variant_valid(x, RELATION_HARD_NEGATIVE))
    return {
        "selected_pos_count": len(selected_pos),
        "selected_neg_count": len(selected_neg),
        "selected_pos_valid_count": selected_pos_valid,
        "selected_neg_valid_count": selected_neg_valid,
        "stats": stats,
        "ok": (
            len(selected_pos) >= cfg.num_hard_pos
            and len(selected_neg) >= cfg.num_hard_neg
            and selected_pos_valid >= cfg.num_hard_pos
            and selected_neg_valid >= cfg.num_hard_neg
        ),
    }


def _combinations(pool: Sequence[Dict], min_k: int) -> Iterable[Tuple[Dict, ...]]:
    if len(pool) < min_k:
        return []
    return itertools.chain.from_iterable(itertools.combinations(pool, k) for k in range(min_k, len(pool) + 1))


def _set_signature(items: Sequence[Dict]) -> Tuple[str, ...]:
    return tuple(sorted(normalize_rewrite_text(x.get("text", "")) for x in items if isinstance(x, dict)))


def _build_candidate_sets(
    final_pos: Sequence[Dict],
    verified_pos_pool: Sequence[Dict],
    cfg: SamplerCfg,
) -> List[List[Dict]]:
    candidates: List[List[Dict]] = []
    seen = set()

    def _add(pos_items: Sequence[Dict]) -> None:
        sig = _set_signature(pos_items)
        if sig in seen:
            return
        seen.add(sig)
        candidates.append([copy.deepcopy(x) for x in pos_items])

    _add(final_pos)
    for comb in _combinations(verified_pos_pool, cfg.num_hard_pos):
        _add(comb)
    return candidates


def _repair_entry(final_entry: Dict, verified_entry: Dict, cfg: SamplerCfg) -> Optional[Dict]:
    anchor = str(final_entry.get("anchor_text", ""))
    final_pos = list(final_entry.get("hard_positives", []))
    final_neg = list(final_entry.get("hard_negatives", []))

    verified_candidates = list(verified_entry.get("verified_candidates", []))
    verified_pos = [x for x in verified_candidates if x.get("relation_label") == RELATION_HARD_POSITIVE]
    verified_neg = [x for x in verified_candidates if x.get("relation_label") == RELATION_HARD_NEGATIVE]

    pos_pool = _dedupe_by_text(list(verified_pos) + list(final_pos), RELATION_HARD_POSITIVE)
    neg_pool = _dedupe_by_text(list(verified_neg) + list(final_neg), RELATION_HARD_NEGATIVE)

    pos_sets = _build_candidate_sets(final_pos=final_pos, verified_pos_pool=pos_pool, cfg=cfg)
    final_pos_sig = _set_signature(final_pos)
    final_neg_sig = _set_signature(final_neg)

    best = None
    best_key = None

    for pos_items in pos_sets:
        pos_sig = _set_signature(pos_items)
        for neg_comb in _combinations(neg_pool, cfg.num_hard_neg):
            neg_items = [copy.deepcopy(x) for x in neg_comb]
            result = _evaluate(anchor_text=anchor, pos_items=pos_items, neg_items=neg_items, cfg=cfg)
            if not result["ok"]:
                continue
            neg_sig = _set_signature(neg_items)
            changed = len(set(pos_sig) ^ set(final_pos_sig)) + len(set(neg_sig) ^ set(final_neg_sig))
            conf_sum = sum(_safe_conf(x) for x in neg_items) + sum(_safe_conf(x) for x in pos_items)
            key = (changed, len(neg_items), len(pos_items), -conf_sum)
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "hard_positives": pos_items,
                    "hard_negatives": neg_items,
                    "post_eval": result,
                }

    if best is None:
        return None
    return best


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Targeted repair for Semantic final cache rows that fail runtime rewrite sampling.")
    parser.add_argument("--final_jsonl", type=str, required=True, help="Path to semantic_perturb_train.jsonl")
    parser.add_argument("--verified_jsonl", type=str, required=True, help="Path to semantic_perturb_train.verified.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="", help="Output path; empty means overwrite final_jsonl")
    parser.add_argument("--report_json", type=str, default="", help="Optional report output path")
    parser.add_argument("--no_backup", action="store_true", help="Disable automatic backup when overwriting input file")
    parser.add_argument("--dry_run", action="store_true", help="Analyze and report only, do not write output")

    parser.add_argument("--num_hard_neg", type=int, default=2)
    parser.add_argument("--num_hard_pos", type=int, default=2)
    parser.add_argument("--risky_negative_overlap_threshold", type=float, default=0.9)
    parser.add_argument("--risky_negative_start_epoch", type=int, default=2)
    parser.add_argument("--risky_negative_downweight", type=float, default=0.5)
    parser.add_argument("--current_epoch", type=int, default=0)

    parser.add_argument("--rewrite_type_quota_enabled", action="store_true", default=True)
    parser.add_argument("--rewrite_type_quota_disabled", action="store_true")
    parser.add_argument("--collision_sanitization_enabled", action="store_true", default=True)
    parser.add_argument("--collision_sanitization_disabled", action="store_true")
    parser.add_argument("--risky_negative_filter_enabled", action="store_true", default=True)
    parser.add_argument("--risky_negative_filter_disabled", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.rewrite_type_quota_disabled:
        args.rewrite_type_quota_enabled = False
    if args.collision_sanitization_disabled:
        args.collision_sanitization_enabled = False
    if args.risky_negative_filter_disabled:
        args.risky_negative_filter_enabled = False

    cfg = SamplerCfg(
        num_hard_neg=int(args.num_hard_neg),
        num_hard_pos=int(args.num_hard_pos),
        collision_sanitization_enabled=bool(args.collision_sanitization_enabled),
        rewrite_type_quota_enabled=bool(args.rewrite_type_quota_enabled),
        risky_negative_filter_enabled=bool(args.risky_negative_filter_enabled),
        risky_negative_overlap_threshold=float(args.risky_negative_overlap_threshold),
        risky_negative_start_epoch=int(args.risky_negative_start_epoch),
        risky_negative_downweight=float(args.risky_negative_downweight),
        current_epoch=int(args.current_epoch),
    )

    final_rows = _load_jsonl(args.final_jsonl)
    verified_rows = _load_jsonl(args.verified_jsonl)
    verified_by_id = {int(x["desc_id"]): x for x in verified_rows if isinstance(x, dict) and "desc_id" in x}

    output_rows: List[Dict] = []
    failing_before: List[Dict] = []
    repaired: List[Dict] = []
    unresolved: List[Dict] = []

    for row in final_rows:
        desc_id = int(row.get("desc_id"))
        anchor = str(row.get("anchor_text", ""))
        pos = list(row.get("hard_positives", []))
        neg = list(row.get("hard_negatives", []))
        pre = _evaluate(anchor_text=anchor, pos_items=pos, neg_items=neg, cfg=cfg)
        pre_load_ok, pre_load_reason = _entry_load_valid(row, cfg)
        if pre["ok"] and pre_load_ok:
            output_rows.append(row)
            continue

        fail_info = {
            "desc_id": desc_id,
            "pre_selected_pos": pre["selected_pos_count"],
            "pre_selected_neg": pre["selected_neg_count"],
            "pre_selected_pos_valid": pre["selected_pos_valid_count"],
            "pre_selected_neg_valid": pre["selected_neg_valid_count"],
            "pre_collision_removed_negative": int(pre["stats"].collision_removed_negative),
            "pre_risky_negative_filtered": int(pre["stats"].risky_negative_filtered),
            "pre_load_valid": bool(pre_load_ok),
            "pre_load_reason": pre_load_reason,
        }
        failing_before.append(fail_info)

        verified_entry = verified_by_id.get(desc_id)
        if verified_entry is None:
            unresolved.append({**fail_info, "reason": "missing_verified_entry"})
            output_rows.append(row)
            continue

        repair = _repair_entry(final_entry=row, verified_entry=verified_entry, cfg=cfg)
        if repair is None:
            unresolved.append({**fail_info, "reason": "no_feasible_subset_from_verified"})
            output_rows.append(row)
            continue

        patched = copy.deepcopy(row)
        patched["hard_positives"] = repair["hard_positives"]
        patched["hard_negatives"] = repair["hard_negatives"]
        post = _evaluate(anchor_text=anchor, pos_items=patched["hard_positives"], neg_items=patched["hard_negatives"], cfg=cfg)
        post_load_ok, post_load_reason = _entry_load_valid(patched, cfg)
        if not post["ok"] or not post_load_ok:
            unresolved.append(
                {
                    **fail_info,
                    "reason": "internal_post_check_failed",
                    "post_ok": bool(post["ok"]),
                    "post_load_valid": bool(post_load_ok),
                    "post_load_reason": post_load_reason,
                }
            )
            output_rows.append(row)
            continue

        repaired.append(
            {
                **fail_info,
                "post_selected_pos": post["selected_pos_count"],
                "post_selected_neg": post["selected_neg_count"],
                "post_selected_pos_valid": post["selected_pos_valid_count"],
                "post_selected_neg_valid": post["selected_neg_valid_count"],
                "post_load_valid": bool(post_load_ok),
                "new_pos_count": len(patched["hard_positives"]),
                "new_neg_count": len(patched["hard_negatives"]),
            }
        )
        output_rows.append(patched)

    failing_after = []
    for row in output_rows:
        desc_id = int(row.get("desc_id"))
        anchor = str(row.get("anchor_text", ""))
        pos = list(row.get("hard_positives", []))
        neg = list(row.get("hard_negatives", []))
        check = _evaluate(anchor_text=anchor, pos_items=pos, neg_items=neg, cfg=cfg)
        check_load_ok, check_load_reason = _entry_load_valid(row, cfg)
        if not check["ok"] or not check_load_ok:
            failing_after.append(
                {
                    "desc_id": desc_id,
                    "selected_pos": check["selected_pos_count"],
                    "selected_neg": check["selected_neg_count"],
                    "selected_pos_valid": check["selected_pos_valid_count"],
                    "selected_neg_valid": check["selected_neg_valid_count"],
                    "collision_removed_negative": int(check["stats"].collision_removed_negative),
                    "risky_negative_filtered": int(check["stats"].risky_negative_filtered),
                    "load_valid": bool(check_load_ok),
                    "load_reason": check_load_reason,
                }
            )

    report = {
        "config": {
            "num_hard_neg": cfg.num_hard_neg,
            "num_hard_pos": cfg.num_hard_pos,
            "collision_sanitization_enabled": cfg.collision_sanitization_enabled,
            "rewrite_type_quota_enabled": cfg.rewrite_type_quota_enabled,
            "risky_negative_filter_enabled": cfg.risky_negative_filter_enabled,
            "risky_negative_overlap_threshold": cfg.risky_negative_overlap_threshold,
            "risky_negative_start_epoch": cfg.risky_negative_start_epoch,
            "risky_negative_downweight": cfg.risky_negative_downweight,
            "current_epoch": cfg.current_epoch,
        },
        "counts": {
            "rows_total": len(final_rows),
            "failing_before": len(failing_before),
            "repaired": len(repaired),
            "unresolved": len(unresolved),
            "failing_after": len(failing_after),
        },
        "failing_before": failing_before,
        "repaired": repaired,
        "unresolved": unresolved,
        "failing_after": failing_after,
    }

    output_path = args.output_jsonl.strip() or args.final_jsonl
    if not args.dry_run:
        if os.path.abspath(output_path) == os.path.abspath(args.final_jsonl) and not args.no_backup:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = args.final_jsonl + ".bak." + ts
            shutil.copy2(args.final_jsonl, backup)
            print("backup_created:", backup)
        _write_jsonl(output_path, output_rows)
        print("written:", output_path)
    else:
        print("dry_run=true, no file written")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("report_written:", args.report_json)

    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
