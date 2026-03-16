import hashlib
import json
import os
import subprocess
from typing import Any, Dict


def sha256_bytes(value: bytes) -> str:
    h = hashlib.sha256()
    h.update(value)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Dict[str, Any]) -> str:
    normalized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(normalized)


def get_git_commit_or_unknown(cwd: str) -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        commit = output.decode("utf-8").strip()
        return commit or "unknown"
    except Exception:
        return "unknown"


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def resolve_output_paths(output_path: str, dataset: str, split: str) -> Dict[str, str]:
    base = output_path.strip() if output_path else ""
    if not base:
        base = os.path.join("method_tvr", "semantic_cache", "{}_{}".format(dataset, split))

    if base.endswith(".jsonl"):
        cache_path = base
        prefix = base[:-6]
    else:
        os.makedirs(base, exist_ok=True)
        cache_path = os.path.join(base, "semantic_cache_{}_{}.jsonl".format(dataset, split))
        prefix = cache_path[:-6]

    return {
        "cache_jsonl": cache_path,
        "manifest_json": prefix + ".manifest.json",
        "summary_json": prefix + ".summary.json",
        "failure_json": prefix + ".failures.json",
        "stats_md": prefix + ".stats.md",
    }
