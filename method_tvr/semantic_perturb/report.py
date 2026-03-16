from collections import Counter
from typing import Dict, List


def build_summary(records: List[Dict], failures: List[Dict], manifest: Dict) -> Dict:
    neg_counter = Counter()
    pos_counter = Counter()
    severity_counter = Counter()

    total_neg = 0
    total_pos = 0
    for item in records:
        for neg in item.get("hard_negatives", []):
            total_neg += 1
            neg_counter[neg["perturbation_type"]] += 1
            severity_counter[str(neg["severity"])] += 1
        for pos in item.get("hard_positives", []):
            total_pos += 1
            pos_counter[pos["perturbation_type"]] += 1
            severity_counter[str(pos["severity"])] += 1

    summary = {
        "num_anchor_total": len(records) + len(failures),
        "num_anchor_success": len(records),
        "num_anchor_failed": len(failures),
        "num_hard_negative_total": total_neg,
        "num_hard_positive_total": total_pos,
        "hard_negative_type_counts": dict(sorted(neg_counter.items())),
        "hard_positive_type_counts": dict(sorted(pos_counter.items())),
        "severity_counts": dict(sorted(severity_counter.items())),
        "manifest": manifest,
    }
    return summary


def summary_to_markdown(summary: Dict) -> str:
    lines = [
        "# Semantic Cache Build Stats",
        "",
        "- anchors_total: {}".format(summary["num_anchor_total"]),
        "- anchors_success: {}".format(summary["num_anchor_success"]),
        "- anchors_failed: {}".format(summary["num_anchor_failed"]),
        "- hard_negative_total: {}".format(summary["num_hard_negative_total"]),
        "- hard_positive_total: {}".format(summary["num_hard_positive_total"]),
        "",
        "## Hard Negative Type Counts",
    ]

    neg_counts = summary.get("hard_negative_type_counts", {})
    if neg_counts:
        for key, value in neg_counts.items():
            lines.append("- {}: {}".format(key, value))
    else:
        lines.append("- <empty>")

    lines.extend(["", "## Hard Positive Type Counts"])
    pos_counts = summary.get("hard_positive_type_counts", {})
    if pos_counts:
        for key, value in pos_counts.items():
            lines.append("- {}: {}".format(key, value))
    else:
        lines.append("- <empty>")

    lines.extend(["", "## Severity Counts"])
    sev_counts = summary.get("severity_counts", {})
    if sev_counts:
        for key, value in sev_counts.items():
            lines.append("- severity_{}: {}".format(key, value))
    else:
        lines.append("- <empty>")

    return "\n".join(lines) + "\n"
