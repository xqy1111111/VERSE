import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DOMINANT_NEGATIVE_TYPES = {"action_swap", "object_scene_swap"}
RISKY_NEGATIVE_TYPES = {"temporal_order_flip", "role_swap"}
PRIORITIZED_POSITIVE_TYPES = {"paraphrase", "modifier_compress"}


def normalize_rewrite_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", str(text or "").lower()))


def compute_text_overlap(anchor_text: str, rewrite_text: str) -> float:
    anchor_tokens = _token_set(anchor_text)
    rewrite_tokens = _token_set(rewrite_text)
    if not anchor_tokens and not rewrite_tokens:
        return 1.0
    if not anchor_tokens or not rewrite_tokens:
        return 0.0
    return float(len(anchor_tokens & rewrite_tokens)) / float(len(anchor_tokens | rewrite_tokens))


@dataclass
class RewriteSamplingStats:
    raw_positive_count: int = 0
    raw_negative_count: int = 0
    valid_positive_count: int = 0
    valid_negative_count: int = 0
    selected_positive_count: int = 0
    selected_negative_count: int = 0
    collision_removed_positive: int = 0
    collision_removed_negative: int = 0
    risky_negative_filtered: int = 0
    risky_negative_downweighted: int = 0
    invalid_removed: int = 0
    positive_type_coverage: int = 0
    negative_type_coverage: int = 0
    non_dominant_negative_available: int = 0
    non_dominant_negative_selected: int = 0

    def to_metrics(self) -> Dict[str, float]:
        return {
            "compositional_raw_positive_count": float(self.raw_positive_count),
            "compositional_raw_negative_count": float(self.raw_negative_count),
            "compositional_valid_positive_count": float(self.valid_positive_count),
            "compositional_valid_negative_count": float(self.valid_negative_count),
            "compositional_selected_positive_count": float(self.selected_positive_count),
            "compositional_selected_negative_count": float(self.selected_negative_count),
            "compositional_collision_removed_positive": float(self.collision_removed_positive),
            "compositional_collision_removed_negative": float(self.collision_removed_negative),
            "compositional_risky_negative_filtered": float(self.risky_negative_filtered),
            "compositional_risky_negative_downweighted": float(self.risky_negative_downweighted),
            "compositional_invalid_removed": float(self.invalid_removed),
            "compositional_positive_type_coverage": float(self.positive_type_coverage),
            "compositional_negative_type_coverage": float(self.negative_type_coverage),
            "compositional_non_dominant_negative_available": float(self.non_dominant_negative_available),
            "compositional_non_dominant_negative_selected": float(self.non_dominant_negative_selected),
        }


def _is_valid_rewrite_item(item: Dict, expected_label: str) -> bool:
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
    verifier = item.get("verifier", {})
    if not isinstance(verifier, dict):
        return False
    if "semantic_relation" not in verifier or "confidence" not in verifier:
        return False
    return True


def _safe_confidence(item: Dict) -> float:
    try:
        return float(item.get("verifier", {}).get("confidence", 0.0))
    except Exception:  # noqa: PERF203
        return 0.0


def _sample_with_type_preference(
    items: Sequence[Dict],
    sample_size: int,
    *,
    preferred_types: Optional[Iterable[str]] = None,
    avoid_types: Optional[Iterable[str]] = None,
    enable_quota: bool = True,
) -> List[Dict]:
    if sample_size <= 0 or not items:
        return []
    ranked = list(items)
    ranked.sort(
        key=lambda x: (
            -_safe_confidence(x),
            int(x.get("severity", 0)),
            str(x.get("perturbation_type", "")),
            normalize_rewrite_text(x.get("text", "")),
        )
    )
    if sample_size >= len(ranked):
        return ranked

    selected: List[Dict] = []
    selected_types = set()
    preferred_types = set(preferred_types or [])
    avoid_types = set(avoid_types or [])

    if enable_quota and preferred_types:
        for item in ranked:
            item_type = str(item.get("perturbation_type", ""))
            if item_type in preferred_types:
                selected.append(item)
                selected_types.add(item_type)
                break

    if enable_quota and avoid_types and len(selected) < sample_size:
        for item in ranked:
            if item in selected:
                continue
            item_type = str(item.get("perturbation_type", ""))
            if item_type not in avoid_types:
                selected.append(item)
                selected_types.add(item_type)
                break

    if len(selected) < sample_size:
        for item in ranked:
            if item in selected:
                continue
            item_type = str(item.get("perturbation_type", ""))
            if enable_quota and item_type not in selected_types:
                selected.append(item)
                selected_types.add(item_type)
            if len(selected) >= sample_size:
                break

    if len(selected) < sample_size:
        for item in ranked:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= sample_size:
                break

    return selected[:sample_size]


def sanitize_and_sample_rewrites(
    *,
    anchor_text: str,
    positive_rewrites: Sequence[Dict],
    negative_rewrites: Sequence[Dict],
    positive_sample_size: int,
    negative_sample_size: int,
    collision_sanitization_enabled: bool,
    rewrite_type_quota_enabled: bool,
    risky_negative_filter_enabled: bool,
    risky_negative_overlap_threshold: float,
    risky_negative_start_epoch: int,
    risky_negative_downweight: float,
    current_epoch: int,
) -> Tuple[List[Dict], List[Dict], List[float], List[float], RewriteSamplingStats]:
    stats = RewriteSamplingStats()
    stats.raw_positive_count = len(positive_rewrites)
    stats.raw_negative_count = len(negative_rewrites)

    valid_pos = [item for item in positive_rewrites if _is_valid_rewrite_item(item, "hard_positive")]
    valid_neg = [item for item in negative_rewrites if _is_valid_rewrite_item(item, "hard_negative")]
    stats.valid_positive_count = len(valid_pos)
    stats.valid_negative_count = len(valid_neg)
    stats.invalid_removed = max(0, stats.raw_positive_count - len(valid_pos)) + max(0, stats.raw_negative_count - len(valid_neg))

    if collision_sanitization_enabled:
        pos_map = defaultdict(list)
        neg_map = defaultdict(list)
        for item in valid_pos:
            pos_map[normalize_rewrite_text(item["text"])].append(item)
        for item in valid_neg:
            neg_map[normalize_rewrite_text(item["text"])].append(item)

        collisions = set(pos_map.keys()) & set(neg_map.keys())
        if collisions:
            # Deterministic policy: keep positive rewrites, drop collided negatives.
            kept_neg = []
            for item in valid_neg:
                if normalize_rewrite_text(item["text"]) in collisions:
                    stats.collision_removed_negative += 1
                    continue
                kept_neg.append(item)
            valid_neg = kept_neg

    candidate_neg_weights = []
    candidate_debias_weights = []
    filtered_neg = []
    for item in valid_neg:
        rewrite_type = str(item.get("perturbation_type", ""))
        semantic_relation = str(item.get("verifier", {}).get("semantic_relation", ""))
        overlap = compute_text_overlap(anchor_text, item.get("text", ""))
        is_risky_type = rewrite_type in RISKY_NEGATIVE_TYPES
        risky_overlap = overlap >= float(risky_negative_overlap_threshold)
        not_contradiction = semantic_relation != "contradiction"
        risky_candidate = bool(is_risky_type and risky_overlap and not_contradiction)

        if risky_candidate and current_epoch < int(risky_negative_start_epoch):
            stats.risky_negative_filtered += 1
            continue
        if risky_candidate and risky_negative_filter_enabled:
            stats.risky_negative_filtered += 1
            continue

        weight = float(max(0.0, min(1.0, _safe_confidence(item))))
        if risky_candidate:
            weight *= float(risky_negative_downweight)
            stats.risky_negative_downweighted += 1

        item_with_meta = dict(item)
        item_with_meta["_rewrite_overlap"] = overlap
        item_with_meta["_rewrite_weight"] = weight
        item_with_meta["_debiased_weight"] = float(max(0.0, min(1.0, overlap)))
        filtered_neg.append(item_with_meta)
        candidate_neg_weights.append(weight)
        candidate_debias_weights.append(item_with_meta["_debiased_weight"])

    non_dominant_available = [x for x in filtered_neg if str(x.get("perturbation_type", "")) not in DOMINANT_NEGATIVE_TYPES]
    stats.non_dominant_negative_available = len(non_dominant_available)

    selected_pos = _sample_with_type_preference(
        valid_pos,
        int(positive_sample_size),
        preferred_types=PRIORITIZED_POSITIVE_TYPES,
        enable_quota=bool(rewrite_type_quota_enabled),
    )
    selected_neg = _sample_with_type_preference(
        filtered_neg,
        int(negative_sample_size),
        avoid_types=DOMINANT_NEGATIVE_TYPES,
        enable_quota=bool(rewrite_type_quota_enabled),
    )

    pos_text_set = set()
    final_pos = []
    for item in selected_pos:
        key = normalize_rewrite_text(item["text"])
        if key in pos_text_set:
            continue
        pos_text_set.add(key)
        final_pos.append(item)

    neg_text_set = set()
    final_neg = []
    final_neg_weights = []
    final_debias_weights = []
    for item in selected_neg:
        key = normalize_rewrite_text(item["text"])
        if key in neg_text_set:
            continue
        neg_text_set.add(key)
        final_neg.append(item)
        final_neg_weights.append(float(item.get("_rewrite_weight", _safe_confidence(item))))
        final_debias_weights.append(float(item.get("_debiased_weight", 0.0)))

    stats.selected_positive_count = len(final_pos)
    stats.selected_negative_count = len(final_neg)
    stats.positive_type_coverage = len({str(x.get("perturbation_type", "")) for x in final_pos})
    stats.negative_type_coverage = len({str(x.get("perturbation_type", "")) for x in final_neg})
    stats.non_dominant_negative_selected = len(
        [x for x in final_neg if str(x.get("perturbation_type", "")) not in DOMINANT_NEGATIVE_TYPES]
    )

    for item in final_pos:
        item.pop("_rewrite_overlap", None)
        item.pop("_rewrite_weight", None)
        item.pop("_debiased_weight", None)
    for item in final_neg:
        item.pop("_rewrite_overlap", None)
        item.pop("_rewrite_weight", None)
        item.pop("_debiased_weight", None)

    return final_pos, final_neg, final_neg_weights, final_debias_weights, stats
