import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

from method_tvr.semantic_perturb.cache_io import (
    append_jsonl,
    iter_jsonl,
    load_existing_sample_hashes,
    load_json,
    make_sample_hash,
    resolve_stage_paths,
    write_json,
)
from method_tvr.semantic_perturb.generator import SemanticGenerator
from method_tvr.semantic_perturb.hashing import get_git_commit_or_unknown, sha256_file, utc_timestamp
from method_tvr.semantic_perturb.llm_backend import build_semantic_backend
from method_tvr.semantic_perturb.prompts import GENERATOR_PROMPT_VERSION, VERIFIER_PROMPT_VERSION
from method_tvr.semantic_perturb.schema import NEG_TYPES, POS_TYPES
from method_tvr.semantic_perturb.stats import build_final_summary, build_stage_summary, summary_to_markdown
from method_tvr.semantic_perturb.verifier import SemanticVerifier


@dataclass
class PerturbConfig:
    perturb_enable: bool = True
    perturb_backend: str = "llm"  # none|llm
    perturb_no_fallback: bool = True
    perturb_strict_mode: bool = True

    perturb_dataset: str = ""
    perturb_split: str = "train"
    perturb_source_path: str = ""
    perturb_work_dir: str = "cache"
    perturb_candidate_path: str = ""
    perturb_verified_path: str = ""
    perturb_final_path: str = ""

    perturb_prompt_version: str = GENERATOR_PROMPT_VERSION
    perturb_schema_version: str = "semantic_schema_v2"
    perturb_generator_model: str = "gpt-4.1-mini"
    perturb_verifier_model: str = "gpt-4.1-mini"
    perturb_temperature: float = 0.1
    perturb_seed: int = 2018

    perturb_num_hard_neg: int = 2
    perturb_num_hard_pos: int = 2
    perturb_neg_types: List[str] = field(default_factory=lambda: sorted(NEG_TYPES))
    perturb_pos_types: List[str] = field(default_factory=lambda: sorted(POS_TYPES))
    perturb_severity_levels: List[int] = field(default_factory=lambda: [1, 2, 3])

    perturb_max_retries_same_backend: int = 2
    perturb_build_workers: int = 4
    perturb_verify_workers: int = 8
    perturb_progress_every: int = 100

    perturb_num_shards: int = 1
    perturb_shard_id: int = 0
    perturb_resume: bool = True
    perturb_skip_existing: bool = True

    perturb_llm_api_base: str = ""
    perturb_llm_api_key: str = ""
    perturb_llm_transport: str = "remote_api"
    perturb_llm_response_mode: str = "json_schema"
    perturb_local_model_name_or_path: str = ""
    perturb_local_device: str = "auto"
    perturb_local_mask_backend: str = "auto"
    perturb_local_max_new_tokens: int = 256

    perturb_retry_stage: str = "build"  # build|verify|export
    perturb_failure_path: str = ""


class PerturbBuildError(Exception):
    pass


def _log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] [semantic-perturb] {}".format(ts, message), flush=True)


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = max(0.0, min(1.0, float(done) / float(total)))
    filled = int(width * ratio)
    return "[{}{}]".format("#" * filled, "-" * (width - filled))


def _format_progress_line(*, stage: str, done: int, total: int, ok: int, fail: int, skipped: int, start_ts: float) -> str:
    elapsed = max(1e-6, time.perf_counter() - start_ts)
    speed = done / elapsed if done > 0 else 0.0
    remaining = max(0, total - done)
    eta_s = (remaining / speed) if speed > 0 else 0.0
    pct = (100.0 * done / total) if total > 0 else 100.0
    return (
        "{} {} {}/{} ({:.1f}%) ok={} fail={} skip={} speed={:.2f} rec/s elapsed={:.1f}m eta={:.1f}m".format(
            stage,
            _progress_bar(done, total),
            done,
            total,
            pct,
            ok,
            fail,
            skipped,
            speed,
            elapsed / 60.0,
            eta_s / 60.0,
        )
    )


def _summarize_failures(failures: List[Dict]) -> str:
    if not failures:
        return "none"
    counter = Counter(str(x.get("error_type", "unknown")) for x in failures)
    parts = ["{}={}".format(k, v) for k, v in sorted(counter.items())]
    return ", ".join(parts)


def _validate_common(cfg: PerturbConfig) -> None:
    if not cfg.perturb_enable:
        return
    if cfg.perturb_backend != "llm":
        raise ValueError("perturb_backend must be llm when perturb_enable=true")
    if not cfg.perturb_no_fallback:
        raise ValueError("perturb_no_fallback must be true (strict no-fallback)")
    if int(cfg.perturb_num_shards) <= 0:
        raise ValueError("perturb_num_shards must be positive")
    if int(cfg.perturb_shard_id) < 0 or int(cfg.perturb_shard_id) >= int(cfg.perturb_num_shards):
        raise ValueError("perturb_shard_id must be in [0, perturb_num_shards)")
    if int(cfg.perturb_build_workers) <= 0 or int(cfg.perturb_verify_workers) <= 0:
        raise ValueError("perturb_build_workers/perturb_verify_workers must be positive")
    if int(cfg.perturb_progress_every) <= 0:
        raise ValueError("perturb_progress_every must be positive")
    if int(cfg.perturb_max_retries_same_backend) < -1:
        raise ValueError("perturb_max_retries_same_backend must be >= -1")
    if cfg.perturb_llm_transport not in {"remote_api", "local_xgrammar"}:
        raise ValueError("perturb_llm_transport must be remote_api|local_xgrammar")
    if cfg.perturb_llm_response_mode not in {"json_schema", "none"}:
        raise ValueError("perturb_llm_response_mode must be json_schema|none")
    if int(cfg.perturb_local_max_new_tokens) <= 0:
        raise ValueError("perturb_local_max_new_tokens must be positive")
    invalid_neg = sorted(set(cfg.perturb_neg_types) - NEG_TYPES)
    invalid_pos = sorted(set(cfg.perturb_pos_types) - POS_TYPES)
    if invalid_neg:
        raise ValueError("Unsupported perturb_neg_types: {}".format(invalid_neg))
    if invalid_pos:
        raise ValueError("Unsupported perturb_pos_types: {}".format(invalid_pos))
    if any(int(x) not in {1, 2, 3} for x in cfg.perturb_severity_levels):
        raise ValueError("perturb_severity_levels only support 1/2/3")


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


def _load_source_records(cfg: PerturbConfig) -> Tuple[List[Tuple[int, Dict]], str]:
    if not os.path.isfile(cfg.perturb_source_path):
        raise FileNotFoundError("perturb_source_path not found: {}".format(cfg.perturb_source_path))
    source_hash = sha256_file(cfg.perturb_source_path)
    with open(cfg.perturb_source_path, "r", encoding="utf-8") as f:
        raw_records = [json.loads(x) for x in f if x.strip()]
    all_records = [_normalize_source_record(x) for x in raw_records]
    shard_records: List[Tuple[int, Dict]] = []
    for idx, rec in enumerate(all_records, start=1):
        if (idx - 1) % int(cfg.perturb_num_shards) != int(cfg.perturb_shard_id):
            continue
        shard_records.append((idx, rec))
    return shard_records, source_hash


def _build_manifest(cfg: PerturbConfig, source_hash: str, stage: str, total_source: int, paths: Dict[str, str]) -> Dict:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return {
        "stage": stage,
        "dataset": cfg.perturb_dataset,
        "split": cfg.perturb_split,
        "source_path": os.path.abspath(cfg.perturb_source_path) if cfg.perturb_source_path else "",
        "source_hash": source_hash,
        "prompt_version": cfg.perturb_prompt_version,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "schema_version": cfg.perturb_schema_version,
        "generator_model": cfg.perturb_generator_model,
        "verifier_model": cfg.perturb_verifier_model,
        "backend": cfg.perturb_backend,
        "llm_transport": cfg.perturb_llm_transport,
        "llm_response_mode": cfg.perturb_llm_response_mode,
        "seed": int(cfg.perturb_seed),
        "num_hard_neg": int(cfg.perturb_num_hard_neg),
        "num_hard_pos": int(cfg.perturb_num_hard_pos),
        "neg_types": sorted(cfg.perturb_neg_types),
        "pos_types": sorted(cfg.perturb_pos_types),
        "severity_levels": sorted(int(x) for x in cfg.perturb_severity_levels),
        "no_fallback": bool(cfg.perturb_no_fallback),
        "strict_mode": bool(cfg.perturb_strict_mode),
        "num_shards": int(cfg.perturb_num_shards),
        "shard_id": int(cfg.perturb_shard_id),
        "resume": bool(cfg.perturb_resume),
        "skip_existing": bool(cfg.perturb_skip_existing),
        "build_workers": int(cfg.perturb_build_workers),
        "verify_workers": int(cfg.perturb_verify_workers),
        "progress_every": int(cfg.perturb_progress_every),
        "timestamp": utc_timestamp(),
        "code_version": get_git_commit_or_unknown(repo_root),
        "paths": paths,
        "total_source": int(total_source),
    }


def _build_backend(cfg: PerturbConfig):
    return build_semantic_backend(
        transport=cfg.perturb_llm_transport,
        api_base=cfg.perturb_llm_api_base or None,
        api_key=cfg.perturb_llm_api_key or None,
        response_mode=cfg.perturb_llm_response_mode,
        local_model_name_or_path=cfg.perturb_local_model_name_or_path,
        local_device=cfg.perturb_local_device,
        local_mask_backend=cfg.perturb_local_mask_backend,
        local_max_new_tokens=cfg.perturb_local_max_new_tokens,
    )


def _dedupe_candidates(items: List[Dict]) -> List[Dict]:
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


def _classify_generator_error(err_text: str) -> str:
    text = err_text.lower()
    if "json" in text or "schema" in text or "unsupported keys" in text:
        return "schema_fail"
    if "http" in text or "urlerror" in text or "timeout" in text:
        return "generator_fail"
    return "generator_fail"


def _classify_verifier_error(err_text: str) -> str:
    text = err_text.lower()
    if "json" in text or "schema" in text:
        return "verifier_schema_fail"
    if "http" in text or "urlerror" in text or "timeout" in text:
        return "verifier_fail"
    return "verifier_fail"


def _select_variants(items: List[Dict], limit: int, kind: str, rng: random.Random) -> List[Dict]:
    if limit <= 0:
        return []
    if len(items) < limit:
        raise PerturbBuildError("Not enough {} variants (got {}, need {})".format(kind, len(items), limit))
    ranked = list(items)
    rng.shuffle(ranked)
    if kind == "hard_negative":
        ranked.sort(key=lambda x: (-int(x["severity"]), x["perturbation_type"], x["text"]))
    else:
        ranked.sort(key=lambda x: (int(x["severity"]), x["perturbation_type"], x["text"]))
    return ranked[:limit]


def _resolve_paths(cfg: PerturbConfig) -> Dict[str, str]:
    paths = resolve_stage_paths(cfg.perturb_work_dir, cfg.perturb_split)
    if cfg.perturb_candidate_path:
        paths["candidate_jsonl"] = cfg.perturb_candidate_path
    if cfg.perturb_verified_path:
        paths["verified_jsonl"] = cfg.perturb_verified_path
    if cfg.perturb_final_path:
        paths["final_jsonl"] = cfg.perturb_final_path
    return paths


def build_cache(cfg: PerturbConfig, target_desc_ids: Optional[Set[int]] = None) -> Dict[str, str]:
    paths = _resolve_paths(cfg)
    if not cfg.perturb_enable:
        _log("build-cache skipped: perturb_enable=false")
        return paths
    _validate_common(cfg)
    shard_records, source_hash = _load_source_records(cfg)
    manifest = _build_manifest(cfg, source_hash=source_hash, stage="build-cache", total_source=len(shard_records), paths=paths)

    existing_hashes = set()
    if cfg.perturb_resume and cfg.perturb_skip_existing:
        existing_hashes = load_existing_sample_hashes(paths["candidate_jsonl"])

    backend = _build_backend(cfg)
    generator = SemanticGenerator(
        backend=backend,
        model_name=cfg.perturb_generator_model,
        temperature=cfg.perturb_temperature,
        neg_types=cfg.perturb_neg_types,
        pos_types=cfg.perturb_pos_types,
        severity_levels=cfg.perturb_severity_levels,
        num_hard_neg=cfg.perturb_num_hard_neg,
        num_hard_pos=cfg.perturb_num_hard_pos,
    )

    to_process: List[Tuple[int, Dict]] = []
    skipped_existing = 0
    for idx, rec in shard_records:
        if target_desc_ids is not None and int(rec["desc_id"]) not in target_desc_ids:
            continue
        sample_hash = make_sample_hash(
            source_hash=source_hash,
            prompt_version=cfg.perturb_prompt_version,
            schema_version=cfg.perturb_schema_version,
            generator_model=cfg.perturb_generator_model,
            verifier_model=cfg.perturb_verifier_model,
            seed=cfg.perturb_seed,
            desc_id=rec["desc_id"],
            anchor_text=rec["desc"],
        )
        if sample_hash in existing_hashes:
            skipped_existing += 1
            continue
        to_process.append((idx, rec))

    failures: List[Dict] = []
    success_count = 0
    started = time.perf_counter()
    _log(
        "build-cache start dataset={} split={} shard={}/{} total={} process={} skipped_existing={} workers={} candidate_path={}".format(
            cfg.perturb_dataset or "<unknown>",
            cfg.perturb_split,
            cfg.perturb_shard_id,
            cfg.perturb_num_shards,
            len(shard_records),
            len(to_process),
            skipped_existing,
            cfg.perturb_build_workers,
            paths["candidate_jsonl"],
        )
    )

    def _one(idx_and_record: Tuple[int, Dict]) -> Dict:
        idx, rec = idx_and_record
        rec_start = time.perf_counter()
        sample_hash = make_sample_hash(
            source_hash=source_hash,
            prompt_version=cfg.perturb_prompt_version,
            schema_version=cfg.perturb_schema_version,
            generator_model=cfg.perturb_generator_model,
            verifier_model=cfg.perturb_verifier_model,
            seed=cfg.perturb_seed,
            desc_id=rec["desc_id"],
            anchor_text=rec["desc"],
        )
        attempts = 0
        infinite_retry = int(cfg.perturb_max_retries_same_backend) == -1
        max_attempts = None if infinite_retry else int(cfg.perturb_max_retries_same_backend) + 1
        last_error = None
        while True:
            attempts += 1
            try:
                neg, pos, anchor_analysis, raw_json = generator.generate_with_analysis(
                    anchor_text=rec["desc"],
                    vid_name=rec.get("vid_name"),
                    duration=rec.get("duration"),
                    ts=rec.get("ts"),
                )
                candidates = _dedupe_candidates(list(neg) + list(pos))
                return {
                    "ok": True,
                    "idx": idx,
                    "desc_id": int(rec["desc_id"]),
                    "elapsed": time.perf_counter() - rec_start,
                    "record": {
                        "sample_hash": sample_hash,
                        "desc_id": int(rec["desc_id"]),
                        "anchor_text": rec["desc"],
                        "source_meta": {
                            "vid_name": rec.get("vid_name"),
                            "ts": rec.get("ts"),
                            "duration": rec.get("duration"),
                            "split": cfg.perturb_split,
                            "source_index": int(idx),
                        },
                        "anchor_analysis": anchor_analysis,
                        "candidates": candidates,
                        "build_meta": {
                            "prompt_version": cfg.perturb_prompt_version,
                            "schema_version": cfg.perturb_schema_version,
                            "generator_model": cfg.perturb_generator_model,
                            "verifier_model": cfg.perturb_verifier_model,
                            "source_hash": source_hash,
                            "timestamp": manifest["timestamp"],
                            "seed": int(cfg.perturb_seed),
                        },
                    },
                }
            except Exception as err:  # noqa: PERF203
                last_error = str(err)
                if not infinite_retry and attempts >= int(max_attempts):
                    break
                if infinite_retry and attempts % 20 == 0:
                    _log("build retry-forever: idx={} desc_id={} attempts={} err={}".format(idx, rec["desc_id"], attempts, last_error[:160]))
        return {
            "ok": False,
            "idx": idx,
            "desc_id": int(rec["desc_id"]),
            "elapsed": time.perf_counter() - rec_start,
            "error": last_error or "unknown error",
            "error_type": _classify_generator_error(last_error or ""),
            "record": rec,
        }

    with ThreadPoolExecutor(max_workers=int(cfg.perturb_build_workers)) as executor:
        futures = [executor.submit(_one, item) for item in to_process]
        for done_i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["ok"]:
                append_jsonl(paths["candidate_jsonl"], [result["record"]])
                success_count += 1
            else:
                failures.append(
                    {
                        "stage": "build-cache",
                        "error_type": result["error_type"],
                        "error": result["error"],
                        "desc_id": int(result["desc_id"]),
                        "source_index": int(result["idx"]),
                        "source_record": result["record"],
                    }
                )
            if done_i == 1 or done_i % int(cfg.perturb_progress_every) == 0 or done_i == len(to_process):
                _log(
                    _format_progress_line(
                        stage="build",
                        done=done_i,
                        total=len(to_process),
                        ok=success_count,
                        fail=len(failures),
                        skipped=skipped_existing,
                        start_ts=started,
                    )
                )

    write_json(paths["build_failure_json"], {"stage": "build-cache", "failures": failures})
    write_json(paths["manifest_json"], manifest)
    summary = build_stage_summary(
        stage="build-cache",
        total_source=len(shard_records),
        processed=len(to_process),
        skipped_existing=skipped_existing,
        success=success_count,
        failed=len(failures),
        manifest=manifest,
    )
    write_json(paths["summary_json"], summary)
    with open(paths["stats_md"], "w", encoding="utf-8") as f:
        f.write(summary_to_markdown(summary))
    _log(
        "build-cache done ok={} fail={} skipped_existing={} failure_breakdown={} outputs={{candidate:{}, failures:{}, manifest:{}}}".format(
            success_count,
            len(failures),
            skipped_existing,
            _summarize_failures(failures),
            paths["candidate_jsonl"],
            paths["build_failure_json"],
            paths["manifest_json"],
        )
    )
    return paths


def verify_cache(cfg: PerturbConfig, target_sample_hashes: Optional[Set[str]] = None) -> Dict[str, str]:
    paths = _resolve_paths(cfg)
    if not cfg.perturb_enable:
        _log("verify-cache skipped: perturb_enable=false")
        return paths
    _validate_common(cfg)
    if not os.path.exists(paths["candidate_jsonl"]):
        raise FileNotFoundError("candidate cache not found: {}".format(paths["candidate_jsonl"]))

    candidate_records = list(iter_jsonl(paths["candidate_jsonl"]))
    if int(cfg.perturb_num_shards) > 1:
        candidate_records = [
            rec
            for rec in candidate_records
            if (int(rec.get("source_meta", {}).get("source_index", 1)) - 1) % int(cfg.perturb_num_shards)
            == int(cfg.perturb_shard_id)
        ]

    existing_hashes = set()
    if cfg.perturb_resume and cfg.perturb_skip_existing:
        existing_hashes = load_existing_sample_hashes(paths["verified_jsonl"])

    to_process: List[Dict] = []
    skipped_existing = 0
    for rec in candidate_records:
        sample_hash = str(rec.get("sample_hash", "")).strip()
        if target_sample_hashes is not None and sample_hash not in target_sample_hashes:
            continue
        if sample_hash in existing_hashes:
            skipped_existing += 1
            continue
        to_process.append(rec)

    backend = _build_backend(cfg)
    verifier = SemanticVerifier(
        backend=backend,
        model_name=cfg.perturb_verifier_model,
        temperature=cfg.perturb_temperature,
    )

    failures: List[Dict] = []
    success_count = 0
    started = time.perf_counter()
    _log(
        "verify-cache start dataset={} split={} shard={}/{} total={} process={} skipped_existing={} workers={} candidate_path={} verified_path={}".format(
            cfg.perturb_dataset or "<unknown>",
            cfg.perturb_split,
            cfg.perturb_shard_id,
            cfg.perturb_num_shards,
            len(candidate_records),
            len(to_process),
            skipped_existing,
            cfg.perturb_verify_workers,
            paths["candidate_jsonl"],
            paths["verified_jsonl"],
        )
    )

    def _one(rec: Dict) -> Dict:
        rec_start = time.perf_counter()
        try:
            verified_candidates = []
            candidates = list(rec.get("candidates", []))
            verify_results = verifier.verify_batch(rec["anchor_text"], candidates)
            for cand, v in zip(candidates, verify_results):
                verified_candidates.append(
                    {
                        "text": cand["text"],
                        "relation_label": cand["relation_label"],
                        "perturbation_type": cand["perturbation_type"],
                        "severity": int(cand["severity"]),
                        "short_rationale": cand["short_rationale"],
                        "verifier": {
                            "verdict": v["verdict"],
                            "semantic_relation": v["semantic_relation"],
                            "confidence": float(v["confidence"]),
                            "reason": v["reason"],
                        },
                        "accept": bool(v["accept"]),
                    }
                )
            return {
                "ok": True,
                "elapsed": time.perf_counter() - rec_start,
                "record": {
                    "sample_hash": rec["sample_hash"],
                    "desc_id": int(rec["desc_id"]),
                    "anchor_text": rec["anchor_text"],
                    "source_meta": rec["source_meta"],
                    "anchor_analysis": rec.get("anchor_analysis", ""),
                    "verified_candidates": verified_candidates,
                    "build_meta": rec["build_meta"],
                },
            }
        except Exception as err:  # noqa: PERF203
            return {
                "ok": False,
                "elapsed": time.perf_counter() - rec_start,
                "error": str(err),
                "error_type": _classify_verifier_error(str(err)),
                "record": rec,
            }

    with ThreadPoolExecutor(max_workers=int(cfg.perturb_verify_workers)) as executor:
        futures = [executor.submit(_one, rec) for rec in to_process]
        for done_i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["ok"]:
                append_jsonl(paths["verified_jsonl"], [result["record"]])
                success_count += 1
            else:
                rec = result["record"]
                failures.append(
                    {
                        "stage": "verify-cache",
                        "error_type": result["error_type"],
                        "error": result["error"],
                        "sample_hash": rec.get("sample_hash", ""),
                        "desc_id": int(rec.get("desc_id", -1)),
                        "candidate_record": rec,
                    }
                )
            if done_i == 1 or done_i % int(cfg.perturb_progress_every) == 0 or done_i == len(to_process):
                _log(
                    _format_progress_line(
                        stage="verify",
                        done=done_i,
                        total=len(to_process),
                        ok=success_count,
                        fail=len(failures),
                        skipped=skipped_existing,
                        start_ts=started,
                    )
                )

    manifest = _build_manifest(
        cfg,
        source_hash=str(candidate_records[0].get("build_meta", {}).get("source_hash", "")) if candidate_records else "",
        stage="verify-cache",
        total_source=len(candidate_records),
        paths=paths,
    )
    write_json(paths["verify_failure_json"], {"stage": "verify-cache", "failures": failures})
    write_json(paths["manifest_json"], manifest)
    summary = build_stage_summary(
        stage="verify-cache",
        total_source=len(candidate_records),
        processed=len(to_process),
        skipped_existing=skipped_existing,
        success=success_count,
        failed=len(failures),
        manifest=manifest,
    )
    write_json(paths["summary_json"], summary)
    with open(paths["stats_md"], "w", encoding="utf-8") as f:
        f.write(summary_to_markdown(summary))
    _log(
        "verify-cache done ok={} fail={} skipped_existing={} failure_breakdown={} outputs={{verified:{}, failures:{}, manifest:{}}}".format(
            success_count,
            len(failures),
            skipped_existing,
            _summarize_failures(failures),
            paths["verified_jsonl"],
            paths["verify_failure_json"],
            paths["manifest_json"],
        )
    )
    return paths


def export_final(cfg: PerturbConfig, target_sample_hashes: Optional[Set[str]] = None) -> Dict[str, str]:
    paths = _resolve_paths(cfg)
    if not cfg.perturb_enable:
        _log("export-final skipped: perturb_enable=false")
        return paths
    _validate_common(cfg)
    if not os.path.exists(paths["verified_jsonl"]):
        raise FileNotFoundError("verified cache not found: {}".format(paths["verified_jsonl"]))

    raw_verified_records = list(iter_jsonl(paths["verified_jsonl"]))
    dedup_reversed: List[Dict] = []
    seen_hashes: Set[str] = set()
    for rec in reversed(raw_verified_records):
        sample_hash = str(rec.get("sample_hash", "")).strip()
        if not sample_hash:
            continue
        if sample_hash in seen_hashes:
            continue
        seen_hashes.add(sample_hash)
        dedup_reversed.append(rec)
    verified_records = list(reversed(dedup_reversed))
    existing_hashes = set()
    if cfg.perturb_resume and cfg.perturb_skip_existing:
        existing_hashes = load_existing_sample_hashes(paths["final_jsonl"])

    export_failures: List[Dict] = []
    final_records: List[Dict] = []
    rng_seed = int(cfg.perturb_seed)
    started = time.perf_counter()
    _log(
        "export-final start dataset={} split={} total_verified_raw={} total_verified_dedup={} existing_final={} final_path={}".format(
            cfg.perturb_dataset or "<unknown>",
            cfg.perturb_split,
            len(raw_verified_records),
            len(verified_records),
            len(existing_hashes),
            paths["final_jsonl"],
        )
    )
    done_i = 0
    for rec in verified_records:
        done_i += 1
        sample_hash = str(rec.get("sample_hash", "")).strip()
        if target_sample_hashes is not None and sample_hash not in target_sample_hashes:
            continue
        if sample_hash in existing_hashes:
            continue
        accepted_neg = [
            x for x in rec.get("verified_candidates", []) if x.get("accept") and x.get("relation_label") == "hard_negative"
        ]
        accepted_pos = [
            x for x in rec.get("verified_candidates", []) if x.get("accept") and x.get("relation_label") == "hard_positive"
        ]
        try:
            local_rng = random.Random(rng_seed + int(rec["desc_id"]))
            neg = _select_variants(accepted_neg, int(cfg.perturb_num_hard_neg), "hard_negative", local_rng)
            pos = _select_variants(accepted_pos, int(cfg.perturb_num_hard_pos), "hard_positive", local_rng)
            final_records.append(
                {
                    "sample_hash": sample_hash,
                    "desc_id": int(rec["desc_id"]),
                    "anchor_text": rec["anchor_text"],
                    "source_meta": rec["source_meta"],
                    "hard_negatives": neg,
                    "hard_positives": pos,
                    "build_meta": rec["build_meta"],
                }
            )
        except Exception as err:  # noqa: PERF203
            export_failures.append(
                {
                    "stage": "export-final",
                    "error_type": "export_fail",
                    "error": str(err),
                    "sample_hash": sample_hash,
                    "desc_id": int(rec.get("desc_id", -1)),
                }
            )
        if done_i == 1 or done_i % int(cfg.perturb_progress_every) == 0 or done_i == len(verified_records):
            _log(
                _format_progress_line(
                    stage="export",
                    done=done_i,
                    total=len(verified_records),
                    ok=len(final_records),
                    fail=len(export_failures),
                    skipped=len(existing_hashes),
                    start_ts=started,
                )
            )

    append_jsonl(paths["final_jsonl"], final_records)
    write_json(paths["export_failure_json"], {"stage": "export-final", "failures": export_failures})
    manifest = _build_manifest(
        cfg,
        source_hash=str(verified_records[0].get("build_meta", {}).get("source_hash", "")) if verified_records else "",
        stage="export-final",
        total_source=len(verified_records),
        paths=paths,
    )
    write_json(paths["manifest_json"], manifest)
    summary = build_final_summary(final_records=final_records, export_failures=export_failures, manifest=manifest)
    write_json(paths["summary_json"], summary)
    with open(paths["stats_md"], "w", encoding="utf-8") as f:
        f.write(summary_to_markdown(summary))
    _log(
        "export-final done ok={} fail={} failure_breakdown={} outputs={{final:{}, failures:{}, manifest:{}}}".format(
            len(final_records),
            len(export_failures),
            _summarize_failures(export_failures),
            paths["final_jsonl"],
            paths["export_failure_json"],
            paths["manifest_json"],
        )
    )
    return paths


def retry_failed(cfg: PerturbConfig) -> Dict[str, str]:
    paths = _resolve_paths(cfg)
    if not cfg.perturb_enable:
        _log("retry-failed skipped: perturb_enable=false")
        return paths
    _validate_common(cfg)
    stage = str(cfg.perturb_retry_stage).strip().lower()
    if stage not in {"build", "verify", "export"}:
        raise ValueError("perturb_retry_stage must be one of: build, verify, export")
    _log("retry-failed start: stage={}".format(stage))

    if stage == "build":
        failure_path = cfg.perturb_failure_path or paths["build_failure_json"]
        if not os.path.exists(failure_path):
            raise FileNotFoundError("build failure file not found: {}".format(failure_path))
        failed = load_json(failure_path).get("failures", [])
        target_desc_ids = {int(x["desc_id"]) for x in failed if "desc_id" in x}
        _log("retry-failed(build): {} samples".format(len(target_desc_ids)))
        return build_cache(cfg, target_desc_ids=target_desc_ids)

    if stage == "verify":
        failure_path = cfg.perturb_failure_path or paths["verify_failure_json"]
        if not os.path.exists(failure_path):
            raise FileNotFoundError("verify failure file not found: {}".format(failure_path))
        failed = load_json(failure_path).get("failures", [])
        target_hashes = {str(x.get("sample_hash", "")).strip() for x in failed if str(x.get("sample_hash", "")).strip()}
        _log("retry-failed(verify): {} samples".format(len(target_hashes)))
        return verify_cache(cfg, target_sample_hashes=target_hashes)

    failure_path = cfg.perturb_failure_path or paths["export_failure_json"]
    if not os.path.exists(failure_path):
        raise FileNotFoundError("export failure file not found: {}".format(failure_path))
    failed = load_json(failure_path).get("failures", [])
    target_hashes = {str(x.get("sample_hash", "")).strip() for x in failed if str(x.get("sample_hash", "")).strip()}
    _log("retry-failed(export): {} samples".format(len(target_hashes)))
    return export_final(cfg, target_sample_hashes=target_hashes)
