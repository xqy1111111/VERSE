import json
import re
from typing import Dict, Iterable, List, Sequence, Tuple

RELATION_HARD_NEGATIVE = "hard_negative"
RELATION_HARD_POSITIVE = "hard_positive"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"

SEM_EQUIVALENT = "equivalent"
SEM_CONTRADICTION = "contradiction"
SEM_CHANGED_CORE = "changed_core_semantics"
SEM_AMBIGUOUS = "ambiguous"

NEG_TYPES = {
    "attribute_swap",
    "action_swap",
    "role_swap",
    "temporal_order_flip",
    "count_state_swap",
    "object_scene_swap",
}

POS_TYPES = {
    "paraphrase",
    "syntax_reorder",
    "modifier_compress",
    "lexical_variation",
}

ALL_PERTURB_TYPES = NEG_TYPES | POS_TYPES


def _assert_type(value, expected_type, field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise ValueError("Field '{}' must be of type {}".format(field_name, expected_type.__name__))


def _assert_no_extra_keys(item: Dict, required_keys: Sequence[str], item_name: str) -> None:
    missing = [k for k in required_keys if k not in item]
    if missing:
        raise ValueError("{} missing required keys: {}".format(item_name, missing))
    extras = sorted([k for k in item.keys() if k not in required_keys])
    if extras:
        raise ValueError("{} has unsupported keys: {}".format(item_name, extras))


def _validate_single_sentence_text(text: str, field_name: str) -> None:
    if not text or not text.strip():
        raise ValueError("{} cannot be empty".format(field_name))
    if "\n" in text or "\r" in text:
        raise ValueError("{} must be a single line sentence".format(field_name))
    stripped = text.strip()
    if len(stripped) < 4:
        raise ValueError("{} is too short".format(field_name))
    if stripped.startswith("{") or stripped.startswith("["):
        raise ValueError("{} appears to be JSON, expected natural language".format(field_name))
    # Ignore punctuation inside quoted spans so quoted text like "Paris!!!"
    # does not inflate sentence-boundary heuristics.
    unquoted = re.sub(r'"[^"\n\r]*"', '""', stripped)
    unquoted = re.sub(r"“[^”\n\r]*”", "“”", unquoted)

    # Count terminal punctuation groups instead of raw characters so emphatic
    # tokens like "!!!" do not get treated as multiple sentences.
    terminal_groups = re.findall(r"[.!?]+", unquoted)
    if len(terminal_groups) > 3:
        raise ValueError("{} appears to include multiple sentences".format(field_name))


def generator_response_schema() -> Dict:
    variant_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "minLength": 4},
            "relation_label": {"type": "string", "enum": [RELATION_HARD_NEGATIVE, RELATION_HARD_POSITIVE]},
            "perturbation_type": {"type": "string", "enum": sorted(ALL_PERTURB_TYPES)},
            "severity": {"type": "integer", "enum": [1, 2, 3]},
            "short_rationale": {"type": "string", "minLength": 4, "maxLength": 240},
        },
        "required": ["text", "relation_label", "perturbation_type", "severity", "short_rationale"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "anchor_analysis": {"type": "string", "minLength": 4, "maxLength": 400},
            "hard_negatives": {"type": "array", "items": variant_schema, "minItems": 0},
            "hard_positives": {"type": "array", "items": variant_schema, "minItems": 0},
        },
        "required": ["anchor_analysis", "hard_negatives", "hard_positives"],
    }


def verifier_response_schema() -> Dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": [VERDICT_PASS, VERDICT_FAIL]},
            "semantic_relation": {
                "type": "string",
                "enum": [SEM_EQUIVALENT, SEM_CONTRADICTION, SEM_CHANGED_CORE, SEM_AMBIGUOUS],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string", "minLength": 1, "maxLength": 400},
        },
        "required": ["verdict", "semantic_relation", "confidence", "reason"],
    }


def parse_strict_json(payload: str) -> Dict:
    if not isinstance(payload, str):
        raise ValueError("Expected JSON string payload")
    stripped = payload.strip()
    if not stripped:
        raise ValueError("Empty JSON payload")
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON payload must be an object")
    return parsed


def validate_generator_item(
    item: Dict,
    allowed_neg_types: Iterable[str],
    allowed_pos_types: Iterable[str],
    allowed_severities: Iterable[int],
) -> Dict:
    _assert_type(item, dict, "variant")
    required = ["text", "relation_label", "perturbation_type", "severity", "short_rationale"]
    _assert_no_extra_keys(item, required, "variant")

    text = item["text"]
    relation_label = item["relation_label"]
    perturb_type = item["perturbation_type"]
    severity = item["severity"]
    rationale = item["short_rationale"]

    _assert_type(text, str, "text")
    _assert_type(relation_label, str, "relation_label")
    _assert_type(perturb_type, str, "perturbation_type")
    _assert_type(severity, int, "severity")
    _assert_type(rationale, str, "short_rationale")

    _validate_single_sentence_text(text, "text")
    _validate_single_sentence_text(rationale, "short_rationale")

    allowed_neg_types = set(allowed_neg_types)
    allowed_pos_types = set(allowed_pos_types)
    allowed_severities = set(int(x) for x in allowed_severities)

    if relation_label not in {RELATION_HARD_NEGATIVE, RELATION_HARD_POSITIVE}:
        raise ValueError("Unsupported relation_label '{}'".format(relation_label))

    if relation_label == RELATION_HARD_NEGATIVE and perturb_type not in allowed_neg_types:
        raise ValueError("hard_negative type '{}' is not enabled".format(perturb_type))
    if relation_label == RELATION_HARD_POSITIVE and perturb_type not in allowed_pos_types:
        raise ValueError("hard_positive type '{}' is not enabled".format(perturb_type))

    if severity not in allowed_severities:
        raise ValueError("severity '{}' is not enabled".format(severity))

    return {
        "text": text.strip(),
        "relation_label": relation_label,
        "perturbation_type": perturb_type,
        "severity": severity,
        "short_rationale": rationale.strip(),
    }


def validate_generator_response(
    payload: Dict,
    allowed_neg_types: Iterable[str],
    allowed_pos_types: Iterable[str],
    allowed_severities: Iterable[int],
    return_anchor_analysis: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    _assert_type(payload, dict, "generator_response")
    required = ["anchor_analysis", "hard_negatives", "hard_positives"]
    _assert_no_extra_keys(payload, required, "generator_response")
    _assert_type(payload["anchor_analysis"], str, "anchor_analysis")
    _validate_single_sentence_text(payload["anchor_analysis"], "anchor_analysis")
    _assert_type(payload["hard_negatives"], list, "hard_negatives")
    _assert_type(payload["hard_positives"], list, "hard_positives")

    hard_negatives = []
    hard_positives = []

    for idx, item in enumerate(payload["hard_negatives"]):
        normalized = validate_generator_item(item, allowed_neg_types, allowed_pos_types, allowed_severities)
        if normalized["relation_label"] != RELATION_HARD_NEGATIVE:
            raise ValueError("hard_negatives[{}] has wrong relation_label '{}'".format(idx, normalized["relation_label"]))
        hard_negatives.append(normalized)

    for idx, item in enumerate(payload["hard_positives"]):
        normalized = validate_generator_item(item, allowed_neg_types, allowed_pos_types, allowed_severities)
        if normalized["relation_label"] != RELATION_HARD_POSITIVE:
            raise ValueError("hard_positives[{}] has wrong relation_label '{}'".format(idx, normalized["relation_label"]))
        hard_positives.append(normalized)

    if return_anchor_analysis:
        return hard_negatives, hard_positives, payload["anchor_analysis"].strip()
    return hard_negatives, hard_positives


def validate_verifier_response(payload: Dict) -> Dict:
    _assert_type(payload, dict, "verifier_response")
    required = ["verdict", "semantic_relation", "confidence", "reason"]
    _assert_no_extra_keys(payload, required, "verifier_response")

    verdict = payload["verdict"]
    semantic_relation = payload["semantic_relation"]
    confidence = payload["confidence"]
    reason = payload["reason"]

    _assert_type(verdict, str, "verdict")
    _assert_type(semantic_relation, str, "semantic_relation")
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    _assert_type(reason, str, "reason")

    if verdict not in {VERDICT_PASS, VERDICT_FAIL}:
        raise ValueError("Unsupported verdict '{}'".format(verdict))
    if semantic_relation not in {SEM_EQUIVALENT, SEM_CONTRADICTION, SEM_CHANGED_CORE, SEM_AMBIGUOUS}:
        raise ValueError("Unsupported semantic_relation '{}'".format(semantic_relation))
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not reason or not str(reason).strip():
        raise ValueError("reason cannot be empty")
    reason_normalized = " ".join(str(reason).split())
    if len(reason_normalized) > 400:
        raise ValueError("reason is too long")

    return {
        "verdict": verdict,
        "semantic_relation": semantic_relation,
        "confidence": confidence,
        "reason": reason_normalized,
    }


def is_verifier_accept(verifier_result: Dict, relation_label: str) -> bool:
    verdict = verifier_result["verdict"]
    semantic_relation = verifier_result["semantic_relation"]

    if verdict != VERDICT_PASS:
        return False
    if semantic_relation == SEM_AMBIGUOUS:
        return False

    if relation_label == RELATION_HARD_POSITIVE:
        return semantic_relation == SEM_EQUIVALENT
    if relation_label == RELATION_HARD_NEGATIVE:
        return semantic_relation in {SEM_CONTRADICTION, SEM_CHANGED_CORE}
    raise ValueError("Unsupported relation_label '{}'".format(relation_label))
