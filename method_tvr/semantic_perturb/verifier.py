from typing import Dict

from method_tvr.semantic_perturb import prompts
from method_tvr.semantic_perturb.llm_backend import LLMBackend
from method_tvr.semantic_perturb.schema import (
    is_verifier_accept,
    parse_strict_json,
    validate_verifier_response,
    verifier_response_schema,
)


class SemanticVerifier:
    def __init__(self, backend: LLMBackend, model_name: str, temperature: float):
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature

    def verify(self, anchor_text: str, candidate: Dict) -> Dict:
        system_prompt = prompts.build_verifier_system_prompt()
        user_prompt = prompts.build_verifier_user_prompt(
            anchor_text=anchor_text,
            candidate_text=candidate["text"],
            relation_label=candidate["relation_label"],
            perturbation_type=candidate["perturbation_type"],
        )
        raw_json = self.backend.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=verifier_response_schema(),
            model=self.model_name,
            temperature=self.temperature,
        )
        parsed = parse_strict_json(raw_json)
        verifier_result = validate_verifier_response(parsed)
        verifier_result["accept"] = bool(is_verifier_accept(verifier_result, candidate["relation_label"]))
        return verifier_result
