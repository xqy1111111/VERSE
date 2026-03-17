from collections import Counter
from typing import Dict, Iterable, List


def _count_types(records: Iterable[Dict], field: str) -> Dict[str, int]:
    counter = Counter()
    for rec in records:
        for item in rec.get(field, []):
            perturb_type = str(item.get("perturbation_type", ""))
            if perturb_type:
                counter[perturb_type] += 1
    return dict(sorted(counter.items()))


def _count_severity(records: Iterable[Dict], field: str) -> Dict[str, int]:
    counter = Counter()
    for rec in records:
        for item in rec.get(field, []):
            sev = item.get("severity")
            if sev is None:
                continue
            counter[str(int(sev))] += 1
    return dict(sorted(counter.items()))


def build_stage_summary(
    *,
    stage: str,
    total_source: int,
    processed: int,
    skipped_existing: int,
    success: int,
    failed: int,
    manifest: Dict,
) -> Dict:
    return {
        "stage": stage,
        "num_source_total": int(total_source),
        "num_processed": int(processed),
        "num_skipped_existing": int(skipped_existing),
        "num_success": int(success),
        "num_failed": int(failed),
        "manifest": manifest,
    }


def build_final_summary(
    *,
    final_records: List[Dict],
    export_failures: List[Dict],
    manifest: Dict,
) -> Dict:
    neg_counts = _count_types(final_records, "hard_negatives")
    pos_counts = _count_types(final_records, "hard_positives")
    sev_counter = Counter()
    for key, value in _count_severity(final_records, "hard_negatives").items():
        sev_counter[key] += value
    for key, value in _count_severity(final_records, "hard_positives").items():
        sev_counter[key] += value

    return {
        "num_anchor_total": len(final_records) + len(export_failures),
        "num_anchor_success": len(final_records),
        "num_anchor_failed": len(export_failures),
        "num_hard_negative_total": sum(neg_counts.values()),
        "num_hard_positive_total": sum(pos_counts.values()),
        "hard_negative_type_counts": neg_counts,
        "hard_positive_type_counts": pos_counts,
        "severity_counts": dict(sorted(sev_counter.items())),
        "manifest": manifest,
    }


def summary_to_markdown(summary: Dict) -> str:
    lines = [
        "# Semantic Perturb Stats",
        "",
        "- stage: {}".format(summary.get("stage", "final")),
    ]
    for key in [
        "num_source_total",
        "num_processed",
        "num_skipped_existing",
        "num_success",
        "num_failed",
        "num_anchor_total",
        "num_anchor_success",
        "num_anchor_failed",
        "num_hard_negative_total",
        "num_hard_positive_total",
    ]:
        if key in summary:
            lines.append("- {}: {}".format(key, summary[key]))

    for title, field in [
        ("Hard Negative Type Counts", "hard_negative_type_counts"),
        ("Hard Positive Type Counts", "hard_positive_type_counts"),
        ("Severity Counts", "severity_counts"),
    ]:
        if field not in summary:
            continue
        lines.extend(["", "## {}".format(title)])
        values = summary.get(field, {})
        if not values:
            lines.append("- <empty>")
        else:
            for k, v in values.items():
                lines.append("- {}: {}".format(k, v))

    return "\n".join(lines) + "\n"

