import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from method_tvr.semantic_perturb.generator import SemanticGenerator
from method_tvr.semantic_perturb.hashing import (
    get_git_commit_or_unknown,
    resolve_output_paths,
    sha256_file,
    utc_timestamp,
)
from method_tvr.semantic_perturb.llm_backend import build_semantic_backend
from method_tvr.semantic_perturb.prompts import GENERATOR_PROMPT_VERSION, VERIFIER_PROMPT_VERSION
from method_tvr.semantic_perturb.report import build_summary, summary_to_markdown
from method_tvr.semantic_perturb.schema import NEG_TYPES, POS_TYPES
from method_tvr.semantic_perturb.verifier import SemanticVerifier
from utils.basic_utils import load_jsonl, save_json


@dataclass
class SemanticBuildConfig:
    dset_name: str
    source_path: str
    cache_split: str = "train"
    output_path: str = ""
    backend: str = "llm"
    strict_mode: bool = True
    no_fallback: bool = True
    num_hard_neg: int = 2
    num_hard_pos: int = 2
    max_retries_same_backend: int = 2
    prompt_version: str = GENERATOR_PROMPT_VERSION
    schema_version: str = "semantic_schema_v1"
    generator_model: str = "gpt-4.1-mini"
    verifier_model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    seed: int = 2018
    neg_types: List[str] = field(default_factory=lambda: sorted(NEG_TYPES))
    pos_types: List[str] = field(default_factory=lambda: sorted(POS_TYPES))
    severity_levels: List[int] = field(default_factory=lambda: [1, 2, 3])
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_transport: str = "remote_api"
    llm_response_mode: str = "json_schema"
    local_model_name_or_path: str = ""
    local_device: str = "auto"
    local_mask_backend: str = "auto"
    local_max_new_tokens: int = 256
    progress_every: int = 20
    slow_record_warn_s: float = 45.0
    build_workers: int = 1


class BuildFailure(Exception):
    pass


class FatalBuildError(Exception):
    """Stop the whole cache build on unrecoverable global errors."""

    pass


def _log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] [semantic-cache] {}".format(ts, message), flush=True)


def _normalize_source_record(raw_data: Dict) -> Dict:
    vid_name = raw_data.get("vid_name") or raw_data.get("video")
    ts = raw_data.get("ts") or raw_data.get("time")
    desc = raw_data.get("desc") or raw_data.get("fig_desc") or raw_data.get("cog_desc")
    duration = raw_data.get("duration")

    if raw_data.get("desc_id") is None:
        raise ValueError("Missing desc_id in source record")
    if vid_name is None or ts is None or desc is None:
        raise ValueError("Missing required fields in source record")

    return {
        "desc_id": int(raw_data["desc_id"]),
        "desc": str(desc),
        "vid_name": str(vid_name),
        "duration": duration,
        "ts": ts,
    }


def _dedupe_variants(items: List[Dict]) -> List[Dict]:
    deduped = []
    seen = set()
    for item in items:
        key = (
            item["text"].strip().lower(),
            item["relation_label"],
            item["perturbation_type"],
            int(item["severity"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _select_variants(items: List[Dict], limit: int, kind: str, rng: random.Random) -> List[Dict]:
    if limit <= 0:
        return []
    if len(items) < limit:
        raise BuildFailure("Not enough {} variants (got {}, need {})".format(kind, len(items), limit))

    ranked = list(items)
    rng.shuffle(ranked)
    if kind == "hard_negative":
        ranked.sort(key=lambda x: (-int(x["severity"]), x["perturbation_type"], x["text"]))
    else:
        ranked.sort(key=lambda x: (int(x["severity"]), x["perturbation_type"], x["text"]))
    return ranked[:limit]


def _validate_build_config(cfg: SemanticBuildConfig) -> None:
    if cfg.backend != "llm":
        raise ValueError("Semantic cache builder only supports backend='llm' for now")
    if cfg.num_hard_neg < 0 or cfg.num_hard_pos < 0:
        raise ValueError("num_hard_neg and num_hard_pos must be >= 0")
    if cfg.max_retries_same_backend < 0:
        raise ValueError("max_retries_same_backend must be >= 0")
    if cfg.temperature < 0:
        raise ValueError("temperature must be >= 0")
    if cfg.llm_transport not in {"remote_api", "local_xgrammar"}:
        raise ValueError("llm_transport must be one of: remote_api, local_xgrammar")
    if cfg.llm_response_mode not in {"json_schema", "none"}:
        raise ValueError("llm_response_mode must be one of: json_schema, none")
    if int(cfg.local_max_new_tokens) <= 0:
        raise ValueError("local_max_new_tokens must be positive")
    if int(cfg.progress_every) <= 0:
        raise ValueError("progress_every must be positive")
    if int(cfg.build_workers) <= 0:
        raise ValueError("build_workers must be positive")
    if float(cfg.slow_record_warn_s) < 0:
        raise ValueError("slow_record_warn_s must be >= 0")
    if cfg.llm_transport == "local_xgrammar" and int(cfg.build_workers) > 1:
        raise ValueError("build_workers > 1 is not supported with local_xgrammar transport")
    if cfg.llm_transport == "local_xgrammar" and not str(cfg.local_model_name_or_path).strip():
        raise ValueError("local_xgrammar transport requires local_model_name_or_path")
    invalid_neg = sorted(set(cfg.neg_types) - NEG_TYPES)
    invalid_pos = sorted(set(cfg.pos_types) - POS_TYPES)
    if invalid_neg:
        raise ValueError("Unsupported neg types: {}".format(invalid_neg))
    if invalid_pos:
        raise ValueError("Unsupported pos types: {}".format(invalid_pos))
    for sev in cfg.severity_levels:
        if int(sev) not in {1, 2, 3}:
            raise ValueError("severity_levels only support 1/2/3")
    if not os.path.isfile(cfg.source_path):
        raise FileNotFoundError("source_path does not exist: {}".format(cfg.source_path))


def _build_manifest(cfg: SemanticBuildConfig, source_hash: str, source_path: str) -> Dict:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        "dataset": cfg.dset_name,
        "split": cfg.cache_split,
        "source_path": os.path.abspath(source_path),
        "source_hash": source_hash,
        "prompt_version": cfg.prompt_version,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "schema_version": cfg.schema_version,
        "generator_model": cfg.generator_model,
        "verifier_model": cfg.verifier_model,
        "backend": cfg.backend,
        "llm_transport": cfg.llm_transport,
        "llm_response_mode": cfg.llm_response_mode,
        "local_model_name_or_path": cfg.local_model_name_or_path,
        "local_device": cfg.local_device,
        "local_mask_backend": cfg.local_mask_backend,
        "local_max_new_tokens": int(cfg.local_max_new_tokens),
        "progress_every": int(cfg.progress_every),
        "build_workers": int(cfg.build_workers),
        "slow_record_warn_s": float(cfg.slow_record_warn_s),
        "seed": cfg.seed,
        "timestamp": utc_timestamp(),
        "code_version": get_git_commit_or_unknown(repo_root),
        "num_hard_neg": cfg.num_hard_neg,
        "num_hard_pos": cfg.num_hard_pos,
        "neg_types": sorted(cfg.neg_types),
        "pos_types": sorted(cfg.pos_types),
        "severity_levels": sorted(int(x) for x in cfg.severity_levels),
        "no_fallback": bool(cfg.no_fallback),
        "strict_mode": bool(cfg.strict_mode),
    }


def _build_one_record(
    *,
    record: Dict,
    generator: SemanticGenerator,
    verifier: SemanticVerifier,
    cfg: SemanticBuildConfig,
    manifest: Dict,
) -> Dict:
    rng = random.Random(cfg.seed + int(record["desc_id"]))
    accepted_neg: List[Dict] = []
    accepted_pos: List[Dict] = []

    max_attempts = cfg.max_retries_same_backend + 1
    attempt = 0
    last_error = None

    while attempt < max_attempts:
        attempt += 1
        try:
            raw_neg, raw_pos, _ = generator.generate(
                anchor_text=record["desc"],
                vid_name=record.get("vid_name"),
                duration=record.get("duration"),
                ts=record.get("ts"),
            )

            accepted_neg = []
            accepted_pos = []

            for candidate in raw_neg + raw_pos:
                verifier_result = verifier.verify(record["desc"], candidate)
                if not verifier_result["accept"]:
                    continue
                variant = {
                    "text": candidate["text"],
                    "relation_label": candidate["relation_label"],
                    "perturbation_type": candidate["perturbation_type"],
                    "severity": int(candidate["severity"]),
                    "short_rationale": candidate["short_rationale"],
                    "verifier": {
                        "verdict": verifier_result["verdict"],
                        "semantic_relation": verifier_result["semantic_relation"],
                        "confidence": float(verifier_result["confidence"]),
                        "reason": verifier_result["reason"],
                    },
                }
                if variant["relation_label"] == "hard_negative":
                    accepted_neg.append(variant)
                elif variant["relation_label"] == "hard_positive":
                    accepted_pos.append(variant)
                else:
                    raise BuildFailure("Unsupported relation_label '{}'".format(variant["relation_label"]))

            accepted_neg = _dedupe_variants(accepted_neg)
            accepted_pos = _dedupe_variants(accepted_pos)

            selected_neg = _select_variants(accepted_neg, cfg.num_hard_neg, "hard_negative", rng)
            selected_pos = _select_variants(accepted_pos, cfg.num_hard_pos, "hard_positive", rng)

            return {
                "desc_id": int(record["desc_id"]),
                "anchor_text": record["desc"],
                "source_meta": {
                    "vid_name": record.get("vid_name"),
                    "ts": record.get("ts"),
                    "duration": record.get("duration"),
                    "split": cfg.cache_split,
                },
                "hard_negatives": selected_neg,
                "hard_positives": selected_pos,
                "build_meta": {
                    "prompt_version": manifest["prompt_version"],
                    "schema_version": manifest["schema_version"],
                    "generator_model": manifest["generator_model"],
                    "verifier_model": manifest["verifier_model"],
                    "source_hash": manifest["source_hash"],
                    "timestamp": manifest["timestamp"],
                    "seed": manifest["seed"],
                },
            }
        except Exception as err:  # noqa: PERF203
            err_text = str(err)
            if "HTTPError 401" in err_text or "HTTPError 403" in err_text:
                raise FatalBuildError(
                    "desc_id={} unrecoverable auth/permission error: {}".format(record["desc_id"], err_text)
                ) from err
            last_error = err

    raise BuildFailure("desc_id={} build failed after {} attempts: {}".format(record["desc_id"], max_attempts, last_error))


def _build_one_record_result(
    *,
    idx: int,
    record: Dict,
    generator: SemanticGenerator,
    verifier: SemanticVerifier,
    cfg: SemanticBuildConfig,
    manifest: Dict,
) -> Dict:
    record_start_ts = time.perf_counter()
    try:
        built = _build_one_record(record=record, generator=generator, verifier=verifier, cfg=cfg, manifest=manifest)
        return {
            "idx": int(idx),
            "record": record,
            "ok": True,
            "elapsed": time.perf_counter() - record_start_ts,
            "built_record": built,
        }
    except FatalBuildError as err:
        return {
            "idx": int(idx),
            "record": record,
            "ok": False,
            "fatal": True,
            "elapsed": time.perf_counter() - record_start_ts,
            "error": str(err),
        }
    except Exception as err:  # noqa: PERF203
        return {
            "idx": int(idx),
            "record": record,
            "ok": False,
            "fatal": False,
            "elapsed": time.perf_counter() - record_start_ts,
            "error": str(err),
        }


def _consume_record_result(
    *,
    result: Dict,
    cfg: SemanticBuildConfig,
    total: int,
    start_ts: float,
    cache_f,
    cache_records: List[Dict],
    failures: List[Dict],
) -> None:
    idx = int(result["idx"])
    record = result["record"]
    ok = bool(result["ok"])

    if ok:
        built_record = result["built_record"]
        cache_records.append(built_record)
        cache_f.write(json.dumps(built_record, ensure_ascii=True) + "\n")
        # Keep the on-disk cache visible and growing during long builds.
        cache_f.flush()
    elif result.get("fatal"):
        err = str(result.get("error", "unknown fatal error"))
        _log("fatal: {}".format(err))
        raise RuntimeError("Semantic cache build aborted: {}".format(err))
    else:
        failures.append(
            {
                "desc_id": int(record["desc_id"]),
                "anchor_text": record["desc"],
                "error": str(result.get("error", "unknown error")),
            }
        )

    record_elapsed = float(result["elapsed"])
    if record_elapsed >= float(cfg.slow_record_warn_s):
        _log(
            "slow-record: idx={}/{} desc_id={} status={} elapsed={:.1f}s".format(
                idx,
                total,
                int(record["desc_id"]),
                "ok" if ok else "fail",
                record_elapsed,
            )
        )

    should_report_progress = (idx == 1) or (idx == total) or (idx % int(cfg.progress_every) == 0)
    if should_report_progress:
        elapsed = max(1e-6, time.perf_counter() - start_ts)
        speed = idx / elapsed
        remaining = total - idx
        eta_s = (remaining / speed) if speed > 0 else 0.0
        _log(
            "progress: {}/{} ({:.1f}%) ok={} fail={} speed={:.2f} rec/s eta={:.1f}m".format(
                idx,
                total,
                100.0 * idx / max(total, 1),
                len(cache_records),
                len(failures),
                speed,
                eta_s / 60.0,
            )
        )


def build_semantic_cache(cfg: SemanticBuildConfig) -> Dict[str, str]:
    _validate_build_config(cfg)

    records = [_normalize_source_record(e) for e in load_jsonl(cfg.source_path)]
    total = len(records)
    source_hash = sha256_file(cfg.source_path)
    manifest = _build_manifest(cfg, source_hash=source_hash, source_path=cfg.source_path)

    out_paths = resolve_output_paths(cfg.output_path, cfg.dset_name, cfg.cache_split)
    out_dir = os.path.dirname(out_paths["cache_jsonl"]) or "."
    os.makedirs(out_dir, exist_ok=True)

    backend = build_semantic_backend(
        transport=cfg.llm_transport,
        api_base=cfg.llm_api_base or None,
        api_key=cfg.llm_api_key or None,
        response_mode=cfg.llm_response_mode,
        local_model_name_or_path=cfg.local_model_name_or_path,
        local_device=cfg.local_device,
        local_mask_backend=cfg.local_mask_backend,
        local_max_new_tokens=cfg.local_max_new_tokens,
    )
    generator = SemanticGenerator(
        backend=backend,
        model_name=cfg.generator_model,
        temperature=cfg.temperature,
        neg_types=cfg.neg_types,
        pos_types=cfg.pos_types,
        severity_levels=cfg.severity_levels,
        num_hard_neg=cfg.num_hard_neg,
        num_hard_pos=cfg.num_hard_pos,
    )
    verifier = SemanticVerifier(backend=backend, model_name=cfg.verifier_model, temperature=cfg.temperature)

    cache_records: List[Dict] = []
    failures: List[Dict] = []
    start_ts = time.perf_counter()
    _log(
        "start: dataset={} split={} total={} transport={} response_mode={} workers={} output={}".format(
            cfg.dset_name,
            cfg.cache_split,
            total,
            cfg.llm_transport,
            cfg.llm_response_mode,
            int(cfg.build_workers),
            out_paths["cache_jsonl"],
        )
    )

    with open(out_paths["cache_jsonl"], "w", encoding="utf-8") as cache_f:
        if int(cfg.build_workers) == 1:
            for idx, record in enumerate(records, start=1):
                result = _build_one_record_result(
                    idx=idx,
                    record=record,
                    generator=generator,
                    verifier=verifier,
                    cfg=cfg,
                    manifest=manifest,
                )
                _consume_record_result(
                    result=result,
                    cfg=cfg,
                    total=total,
                    start_ts=start_ts,
                    cache_f=cache_f,
                    cache_records=cache_records,
                    failures=failures,
                )
        else:
            ordered_results: Dict[int, Dict] = {}
            next_idx = 1
            with ThreadPoolExecutor(max_workers=int(cfg.build_workers)) as executor:
                future_to_idx = {
                    executor.submit(
                        _build_one_record_result,
                        idx=idx,
                        record=record,
                        generator=generator,
                        verifier=verifier,
                        cfg=cfg,
                        manifest=manifest,
                    ): idx
                    for idx, record in enumerate(records, start=1)
                }
                try:
                    for future in as_completed(future_to_idx):
                        result = future.result()
                        if result.get("fatal"):
                            for pending in future_to_idx:
                                if not pending.done():
                                    pending.cancel()
                            err = str(result.get("error", "unknown fatal error"))
                            _log("fatal: {}".format(err))
                            raise RuntimeError("Semantic cache build aborted: {}".format(err))
                        ordered_results[int(result["idx"])] = result
                        while next_idx in ordered_results:
                            ready = ordered_results.pop(next_idx)
                            _consume_record_result(
                                result=ready,
                                cfg=cfg,
                                total=total,
                                start_ts=start_ts,
                                cache_f=cache_f,
                                cache_records=cache_records,
                                failures=failures,
                            )
                            next_idx += 1
                except RuntimeError:
                    for pending in future_to_idx:
                        if not pending.done():
                            pending.cancel()
                    raise

    save_json(manifest, out_paths["manifest_json"], save_pretty=True, sort_keys=True)
    summary = build_summary(cache_records, failures, manifest)
    save_json(summary, out_paths["summary_json"], save_pretty=True, sort_keys=True)
    save_json({"failures": failures}, out_paths["failure_json"], save_pretty=True, sort_keys=True)
    with open(out_paths["stats_md"], "w", encoding="utf-8") as f:
        f.write(summary_to_markdown(summary))

    total_elapsed = max(1e-6, time.perf_counter() - start_ts)
    _log(
        "done: ok={} fail={} total={} elapsed={:.1f}m outputs={}".format(
            len(cache_records),
            len(failures),
            total,
            total_elapsed / 60.0,
            out_paths,
        )
    )

    if cfg.strict_mode and failures:
        _log("strict-mode failure: {} records failed, raising RuntimeError".format(len(failures)))
        raise RuntimeError(
            "Semantic cache build failed in strict mode: {} failures. "
            "See {}".format(len(failures), out_paths["failure_json"])
        )

    return out_paths
