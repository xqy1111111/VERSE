from typing import Dict, Iterable, List, Optional, Tuple

from method_tvr.semantic_perturb import prompts
from method_tvr.semantic_perturb.llm_backend import LLMBackend
from method_tvr.semantic_perturb.schema import (
    generator_response_schema,
    parse_strict_json,
    validate_generator_response,
)


class SemanticGenerator:
    def __init__(
        self,
        backend: LLMBackend,
        model_name: str,
        temperature: float,
        neg_types: Iterable[str],
        pos_types: Iterable[str],
        severity_levels: Iterable[int],
        num_hard_neg: int,
        num_hard_pos: int,
    ):
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.neg_types = list(neg_types)
        self.pos_types = list(pos_types)
        self.severity_levels = list(severity_levels)
        self.num_hard_neg = int(num_hard_neg)
        self.num_hard_pos = int(num_hard_pos)

    def generate(
        self,
        *,
        anchor_text: str,
        vid_name: Optional[str],
        duration: Optional[float],
        ts,
    ) -> Tuple[List[Dict], List[Dict], str]:
        system_prompt = prompts.build_generator_system_prompt()
        user_prompt = prompts.build_generator_user_prompt(
            anchor_text=anchor_text,
            vid_name=vid_name,
            duration=duration,
            ts=ts,
            neg_types=self.neg_types,
            pos_types=self.pos_types,
            severity_levels=self.severity_levels,
            num_hard_neg=self.num_hard_neg,
            num_hard_pos=self.num_hard_pos,
        )
        raw_json = self.backend.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=generator_response_schema(),
            model=self.model_name,
            temperature=self.temperature,
        )
        parsed = parse_strict_json(raw_json)
        hard_negatives, hard_positives = validate_generator_response(
            parsed,
            allowed_neg_types=self.neg_types,
            allowed_pos_types=self.pos_types,
            allowed_severities=self.severity_levels,
        )
        return hard_negatives, hard_positives, raw_json
