from typing import Iterable, Optional


GENERATOR_PROMPT_VERSION = "semantic_generator_v1"
VERIFIER_PROMPT_VERSION = "semantic_verifier_v1"


def build_generator_system_prompt() -> str:
    return (
        "You generate compositional semantic perturbations for video-text retrieval training. "
        "Always return valid JSON matching the provided schema. "
        "Never output markdown, explanations, or non-JSON text."
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
    return (
        "Anchor query: {anchor}\n"
        "Video name (optional context): {vid_name}\n"
        "Duration (optional context): {duration}\n"
        "Timestamp (optional context): {ts}\n"
        "Generate at least {num_hard_neg} hard negatives and {num_hard_pos} hard positives.\n"
        "Hard negatives must use types from: {neg_types}.\n"
        "Hard positives must use types from: {pos_types}.\n"
        "Allowed severity levels: {severity_levels}.\n"
        "Each candidate must be a single natural sentence and include a short rationale sentence.\n"
        "Do not use placeholders or malformed text."
    ).format(
        anchor=anchor_text,
        vid_name=vid_name if vid_name is not None else "<none>",
        duration=duration if duration is not None else "<none>",
        ts=ts if ts is not None else "<none>",
        num_hard_neg=num_hard_neg,
        num_hard_pos=num_hard_pos,
        neg_types=", ".join(sorted(neg_types)),
        pos_types=", ".join(sorted(pos_types)),
        severity_levels=", ".join(str(x) for x in sorted(severity_levels)),
    )


def build_verifier_system_prompt() -> str:
    return (
        "You verify semantic relation between an anchor query and a candidate perturbation. "
        "Always return strict JSON only, using the provided schema."
    )


def build_verifier_user_prompt(anchor_text: str, candidate_text: str, relation_label: str, perturbation_type: str) -> str:
    return (
        "Anchor query: {anchor}\n"
        "Candidate query: {candidate}\n"
        "Expected relation label: {relation_label}\n"
        "Perturbation type: {perturbation_type}\n"
        "Return verdict, semantic_relation, confidence, and reason."
    ).format(
        anchor=anchor_text,
        candidate=candidate_text,
        relation_label=relation_label,
        perturbation_type=perturbation_type,
    )
