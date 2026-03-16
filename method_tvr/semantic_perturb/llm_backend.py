import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any, List


class LLMBackend(ABC):
    @abstractmethod
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        model: str,
        temperature: float,
    ) -> str:
        raise NotImplementedError


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _log(channel: str, message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] [{}] {}".format(ts, channel, message), flush=True)


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible chat completion backend."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        timeout_s: int = 120,
        response_mode: str = "json_schema",
    ):
        if not api_key:
            raise ValueError("Semantic LLM backend requires an API key")
        if response_mode not in {"json_schema", "none"}:
            raise ValueError("Unsupported response_mode '{}', expected one of: json_schema, none".format(response_mode))
        timeout_override = os.environ.get("SEMANTIC_LLM_TIMEOUT_S")
        if timeout_override is not None and str(timeout_override).strip():
            timeout_s = int(float(timeout_override))
        if int(timeout_s) <= 0:
            raise ValueError("timeout_s must be positive")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_s = int(timeout_s)
        self.response_mode = response_mode
        self.log_requests = _env_flag("SEMANTIC_LOG_LLM_REQUESTS", False)
        self.slow_request_warn_s = float(os.environ.get("SEMANTIC_LLM_SLOW_WARN_S", "20"))
        self.http_max_retries = int(os.environ.get("SEMANTIC_LLM_HTTP_MAX_RETRIES", "2"))
        self.http_backoff_s = float(os.environ.get("SEMANTIC_LLM_HTTP_BACKOFF_S", "2.0"))
        self.retryable_status_codes = {429, 500, 502, 503, 504}

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    @staticmethod
    def _extract_text_content(message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            if chunks:
                return "\n".join(chunks)
        raise RuntimeError("LLM response content must be a string")

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        model: str,
        temperature: float,
    ) -> str:
        req_start = time.perf_counter()
        user_prompt_text = user_prompt
        if self.response_mode == "none":
            # Keep remote providers compatible while still requesting strict JSON output.
            user_prompt_text = user_prompt + "\n\nReturn JSON only. Do not add markdown or explanations."

        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_text},
            ],
        }
        if self.response_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_payload",
                    "strict": True,
                    "schema": schema,
                },
            }

        url = self.api_base + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.api_key),
            },
            method="POST",
        )
        body = None
        last_error = None
        for attempt_idx in range(self.http_max_retries + 1):
            if self.log_requests:
                _log(
                    "semantic-llm",
                    "request-start model={} mode={} attempt={}/{} timeout={}s".format(
                        model,
                        self.response_mode,
                        attempt_idx + 1,
                        self.http_max_retries + 1,
                        self.timeout_s,
                    ),
                )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as err:
                last_error = err
                elapsed = time.perf_counter() - req_start
                detail = err.read().decode("utf-8", errors="replace")
                should_retry = attempt_idx < self.http_max_retries and err.code in self.retryable_status_codes
                _log(
                    "semantic-llm",
                    "request-http-error model={} mode={} elapsed={:.1f}s code={} retry={}".format(
                        model, self.response_mode, elapsed, err.code, should_retry
                    ),
                )
                if not should_retry:
                    raise RuntimeError("LLM HTTPError {}: {}".format(err.code, detail)) from err
                backoff = self.http_backoff_s * (2 ** attempt_idx) + random.uniform(0.0, 0.5)
                if self.log_requests:
                    _log("semantic-llm", "retry-backoff {:.1f}s".format(max(0.0, backoff)))
                time.sleep(max(0.0, backoff))
            except urllib.error.URLError as err:
                last_error = err
                elapsed = time.perf_counter() - req_start
                should_retry = attempt_idx < self.http_max_retries
                _log(
                    "semantic-llm",
                    "request-url-error model={} mode={} elapsed={:.1f}s err={} retry={}".format(
                        model, self.response_mode, elapsed, err, should_retry
                    ),
                )
                if not should_retry:
                    raise RuntimeError("LLM URLError: {}".format(err)) from err
                backoff = self.http_backoff_s * (2 ** attempt_idx) + random.uniform(0.0, 0.5)
                if self.log_requests:
                    _log("semantic-llm", "retry-backoff {:.1f}s".format(max(0.0, backoff)))
                time.sleep(max(0.0, backoff))
            except (TimeoutError, socket.timeout) as err:
                last_error = err
                elapsed = time.perf_counter() - req_start
                should_retry = attempt_idx < self.http_max_retries
                _log(
                    "semantic-llm",
                    "request-timeout model={} mode={} elapsed={:.1f}s retry={}".format(
                        model, self.response_mode, elapsed, should_retry
                    ),
                )
                if not should_retry:
                    raise RuntimeError("LLM timeout: {}".format(err)) from err
                backoff = self.http_backoff_s * (2 ** attempt_idx) + random.uniform(0.0, 0.5)
                if self.log_requests:
                    _log("semantic-llm", "retry-backoff {:.1f}s".format(max(0.0, backoff)))
                time.sleep(max(0.0, backoff))

        if body is None:
            raise RuntimeError("LLM request failed after retries: {}".format(last_error))

        elapsed = time.perf_counter() - req_start
        if self.log_requests or elapsed >= self.slow_request_warn_s:
            _log(
                "semantic-llm",
                "request-done model={} mode={} elapsed={:.1f}s".format(model, self.response_mode, elapsed),
            )

        parsed = json.loads(body)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response missing choices")
        message = choices[0].get("message", {})
        content = self._extract_text_content(message).strip()
        if not content:
            raise RuntimeError("LLM response content is empty")
        if self.response_mode == "none":
            maybe_json = self._extract_first_json_object(content)
            if maybe_json is not None:
                return maybe_json
        return content


class LocalXGrammarBackend(LLMBackend):
    """Local HF backend with xgrammar-constrained generation."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        mask_backend: str = "auto",
        max_new_tokens: int = 256,
    ):
        if not model_name_or_path:
            raise ValueError("Local xgrammar backend requires model_name_or_path")
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.model_name_or_path = str(model_name_or_path)
        self.max_new_tokens = int(max_new_tokens)
        self._device = self._resolve_device(device)
        if mask_backend == "auto":
            self.mask_backend = "cuda" if self._device.type == "cuda" else "cpu"
        else:
            self.mask_backend = str(mask_backend)

        try:
            import xgrammar as xgr
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "failed to import local xgrammar dependencies (xgrammar/transformers/torch): {}: {}".format(
                    type(exc).__name__, exc
                )
            ) from exc

        self._xgr = xgr
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        model_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            dtype=model_dtype,
        )
        self._model = self._model.to(self._device).eval()
        if self._model.generation_config is not None:
            self._model.generation_config.do_sample = False
            self._model.generation_config.temperature = 1.0
            self._model.generation_config.top_p = 1.0
            self._model.generation_config.top_k = 50
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._schema_grammar_cache: Dict[str, Any] = {}
        self.log_requests = _env_flag("SEMANTIC_LOG_LLM_REQUESTS", False)
        self.slow_request_warn_s = float(os.environ.get("SEMANTIC_LLM_SLOW_WARN_S", "20"))

    @staticmethod
    def _resolve_device(device: str):
        import torch

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _get_compiled_grammar(self, schema: Dict[str, Any]):
        canonical = json.dumps(schema, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if canonical in self._schema_grammar_cache:
            return self._schema_grammar_cache[canonical]
        tokenizer_info = self._xgr.TokenizerInfo.from_huggingface(self._tokenizer, vocab_size=len(self._tokenizer))
        compiler = self._xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_json_schema(canonical)
        self._schema_grammar_cache[canonical] = compiled
        return compiled

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        model: str,
        temperature: float,
    ) -> str:
        req_start = time.perf_counter()
        _ = model
        _ = temperature
        compiled_grammar = self._get_compiled_grammar(schema)
        prompt = "{}\n\n{}\n\nReturn JSON only.".format(system_prompt.strip(), user_prompt.strip())
        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self._device) for k, v in model_inputs.items()}
        prompt_len = int(model_inputs["input_ids"].shape[1])
        logits_processor = _XGrammarLogitsProcessor(self._xgr, compiled_grammar, self.mask_backend)
        output = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            logits_processor=[logits_processor],
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - req_start
        if self.log_requests or elapsed >= self.slow_request_warn_s:
            _log(
                "semantic-llm",
                "local-request-done model={} elapsed={:.1f}s".format(self.model_name_or_path, elapsed),
            )
        parsed = self._extract_first_json_object(decoded)
        if parsed is None:
            raise RuntimeError("Local xgrammar backend did not produce a JSON object")
        return parsed


class _XGrammarLogitsProcessor:
    """HF logits processor with xgrammar token mask."""

    def __init__(self, xgr_module: Any, compiled_grammar: Any, mask_backend: str):
        self.xgr = xgr_module
        self.mask_backend = mask_backend
        self.matchers: List[Any] = []
        self.compiled_grammars: List[Any] = (
            compiled_grammar if isinstance(compiled_grammar, list) else [compiled_grammar]
        )
        self.full_vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0

    def __call__(self, input_ids, scores):
        if len(self.matchers) == 0:
            self.batch_size = input_ids.shape[0]
            self.compiled_grammars = (
                self.compiled_grammars if len(self.compiled_grammars) > 1 else self.compiled_grammars * self.batch_size
            )
            if len(self.compiled_grammars) != self.batch_size:
                raise RuntimeError("compiled_grammars size must match batch size")
            self.matchers = [self.xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)]
            self.token_bitmask = self.xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)

        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError("input batch mismatch: got {}, expected {}".format(input_ids.shape[0], self.batch_size))

        if not self.prefilled:
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = input_ids[i][-1]
                    if not self.matchers[i].accept_token(sampled_token):
                        raise RuntimeError("xgrammar matcher rejected sampled token")

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        self.xgr.apply_token_bitmask_inplace(
            scores,
            self.token_bitmask.to(scores.device),
            backend=self.mask_backend,
        )
        return scores


class StaticJSONBackend(LLMBackend):
    """Test-only backend that returns predefined JSON strings."""

    def __init__(self, generator_response: str, verifier_response: str):
        self.generator_response = generator_response
        self.verifier_response = verifier_response

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        model: str,
        temperature: float,
    ) -> str:
        prompt = (system_prompt + "\n" + user_prompt).lower()
        if "verify semantic relation" in prompt:
            return self.verifier_response
        return self.generator_response


def build_llm_backend(api_base: Optional[str] = None, api_key: Optional[str] = None) -> LLMBackend:
    api_base = api_base or os.environ.get("SEMANTIC_LLM_API_BASE") or os.environ.get("SILICONFLOW_API_BASE") or "https://api.openai.com/v1"
    api_key = api_key or os.environ.get("SEMANTIC_LLM_API_KEY") or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return OpenAICompatibleBackend(api_base=api_base, api_key=api_key)


def build_semantic_backend(
    *,
    transport: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    response_mode: str = "json_schema",
    local_model_name_or_path: str = "",
    local_device: str = "auto",
    local_mask_backend: str = "auto",
    local_max_new_tokens: int = 256,
) -> LLMBackend:
    transport_name = str(transport or "remote_api")
    if transport_name == "remote_api":
        remote_api_base = (
            api_base
            or os.environ.get("SEMANTIC_LLM_API_BASE")
            or os.environ.get("SILICONFLOW_API_BASE")
            or "https://api.openai.com/v1"
        )
        remote_api_key = (
            api_key
            or os.environ.get("SEMANTIC_LLM_API_KEY")
            or os.environ.get("SILICONFLOW_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        return OpenAICompatibleBackend(
            api_base=remote_api_base,
            api_key=remote_api_key,
            response_mode=response_mode,
        )
    if transport_name == "local_xgrammar":
        model_name = (
            local_model_name_or_path
            or os.environ.get("SEMANTIC_LOCAL_MODEL_NAME_OR_PATH")
            or os.environ.get("LOCAL_MODEL_NAME_OR_PATH")
            or ""
        )
        return LocalXGrammarBackend(
            model_name_or_path=model_name,
            device=local_device,
            mask_backend=local_mask_backend,
            max_new_tokens=local_max_new_tokens,
        )
    raise ValueError("Unsupported transport '{}', expected remote_api|local_xgrammar".format(transport_name))
