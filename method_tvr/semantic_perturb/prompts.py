from typing import Iterable, Optional


GENERATOR_PROMPT_VERSION = "semantic_generator_v1"
VERIFIER_PROMPT_VERSION = "semantic_verifier_v1"


def build_generator_system_prompt() -> str:
    return (
        "You are a high-precision semantic perturbation generator for video-text retrieval training. "
        "Output exactly one JSON object that strictly follows the provided schema. "
        "Never output markdown, code fences, commentary, or any text outside JSON. "
        "Use only natural language sentences in candidate text fields."
    )


def build_generator_user_prompt(
    anchor_text: str,
    vid_name: Optional[str],
    duration: Optional[float],
    ts,
    neg_types: Iterable[str],
    pos_types: Iterable[str],
    severity_levels: Iterable[int],
    num_hard_neg: int,
    num_hard_pos: int,
) -> str:
    neg_types_sorted = sorted(neg_types)
    pos_types_sorted = sorted(pos_types)
    severity_sorted = sorted(int(x) for x in severity_levels)

    # Ask for mild over-generation so verifier/export stages retain enough valid samples.
    target_neg = max(int(num_hard_neg) + 2, int(num_hard_neg))
    target_pos = max(int(num_hard_pos) + 1, int(num_hard_pos))
    target_neg_type_coverage = min(len(neg_types_sorted), target_neg)
    target_pos_type_coverage = min(len(pos_types_sorted), target_pos)

    return (
        "Anchor query: {anchor}\n"
        "Video name (optional context): {vid_name}\n"
        "Duration (optional context): {duration}\n"
        "Timestamp (optional context): {ts}\n"
        "Task:\n"
        "1) Write one concise anchor_analysis sentence describing the core event semantics.\n"
        "2) Generate at least {target_neg} hard negatives and {target_pos} hard positives in one response.\n"
        "3) Hard negatives must use only these types: {neg_types}.\n"
        "4) Hard positives must use only these types: {pos_types}.\n"
        "5) Use only these severity values: {severity_levels}.\n"
        "6) Diversity target: hard_negatives should cover at least {neg_type_cov} distinct negative types; "
        "hard_positives should cover at least {pos_type_cov} distinct positive types when possible.\n"
        "7) Each candidate text must be exactly one natural sentence (single line, no list markers, no placeholders).\n"
        "8) Each short_rationale must be one short sentence that explains why relation/type is valid.\n"
        "9) Do not duplicate candidate text (case-insensitive) across negatives/positives.\n"
        "10) Relation correctness: hard_positive must preserve core semantics; hard_negative must alter core semantics.\n"
        "Output requirements:\n"
        "- Return only valid JSON matching the schema.\n"
        "- Keep wording concrete and fluent, avoid malformed or template-like text."
    ).format(
        anchor=anchor_text,
        vid_name=vid_name if vid_name is not None else "<none>",
        duration=duration if duration is not None else "<none>",
        ts=ts if ts is not None else "<none>",
        target_neg=target_neg,
        target_pos=target_pos,
        neg_types=", ".join(neg_types_sorted),
        pos_types=", ".join(pos_types_sorted),
        severity_levels=", ".join(str(x) for x in severity_sorted),
        neg_type_cov=target_neg_type_coverage,
        pos_type_cov=target_pos_type_coverage,
    )


def build_verifier_system_prompt() -> str:
    return (
        "You are a strict semantic relation verifier. "
        "Return exactly one JSON object following the provided schema. "
        "No markdown or extra text. "
        "Policy: for hard_positive, only equivalent can pass; "
        "for hard_negative, only contradiction or changed_core_semantics can pass; "
        "ambiguous must fail."
    )


def build_verifier_user_prompt(anchor_text: str, candidate_text: str, relation_label: str, perturbation_type: str) -> str:
    return (
        "Anchor query: {anchor}\n"
        "Candidate query: {candidate}\n"
        "Expected relation label: {relation_label}\n"
        "Perturbation type: {perturbation_type}\n"
        "Decision rubric:\n"
        "- semantic_relation must be one of: equivalent, contradiction, changed_core_semantics, ambiguous.\n"
        "- If relation_label=hard_positive, pass only when semantic_relation=equivalent.\n"
        "- If relation_label=hard_negative, pass only when semantic_relation is contradiction or changed_core_semantics.\n"
        "- If semantic_relation=ambiguous, verdict must be fail.\n"
        "- confidence must be in [0, 1].\n"
        "- reason must be one short sentence.\n"
        "Return only verdict, semantic_relation, confidence, reason in strict JSON."
    ).format(
        anchor=anchor_text,
        candidate=candidate_text,
        relation_label=relation_label,
        perturbation_type=perturbation_type,
    )
