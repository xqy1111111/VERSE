import json
import os
from typing import Dict, Iterable, Iterator, List, Set

from method_tvr.semantic_perturb.hashing import sha256_text


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_stage_paths(work_dir: str, split: str) -> Dict[str, str]:
    root = work_dir.strip() if str(work_dir).strip() else "cache"
    os.makedirs(root, exist_ok=True)
    prefix = os.path.join(root, "semantic_perturb_{}".format(split))
    return {
        "candidate_jsonl": prefix + ".candidates.jsonl",
        "verified_jsonl": prefix + ".verified.jsonl",
        "final_jsonl": prefix + ".jsonl",
        "build_failure_json": prefix + ".build.failures.json",
        "verify_failure_json": prefix + ".verify.failures.json",
        "export_failure_json": prefix + ".export.failures.json",
        "manifest_json": prefix + ".manifest.json",
        "summary_json": prefix + ".summary.json",
        "stats_md": prefix + ".stats.md",
    }


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as err:  # noqa: PERF203
                raise RuntimeError("Invalid JSONL at {}:{}: {}".format(path, line_no, err)) from err
            if not isinstance(obj, dict):
                raise RuntimeError("JSONL object at {}:{} must be dict".format(path, line_no))
            yield obj


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    return list(iter_jsonl(path))


def append_jsonl(path: str, records: Iterable[Dict]) -> int:
    ensure_parent_dir(path)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            count += 1
    return count


def write_json(path: str, payload: Dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError("JSON at {} must be object".format(path))
    return payload


def load_existing_sample_hashes(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    sample_hashes: Set[str] = set()
    for rec in iter_jsonl(path):
        sample_hash = str(rec.get("sample_hash", "")).strip()
        if not sample_hash:
            continue
        sample_hashes.add(sample_hash)
    return sample_hashes


def make_sample_hash(
    *,
    source_hash: str,
    prompt_version: str,
    schema_version: str,
    generator_model: str,
    verifier_model: str,
    seed: int,
    desc_id: int,
    anchor_text: str,
) -> str:
    payload = "|".join(
        [
            str(source_hash),
            str(prompt_version),
            str(schema_version),
            str(generator_model),
            str(verifier_model),
            str(seed),
            str(desc_id),
            str(anchor_text).strip(),
        ]
    )
    return sha256_text(payload)

