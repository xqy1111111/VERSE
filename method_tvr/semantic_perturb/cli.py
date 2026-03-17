import argparse
import json
from dataclasses import asdict

from method_tvr.semantic_perturb.builder import PerturbConfig, build_cache, export_final, retry_failed, verify_cache


COMMON_OVERRIDE_KEYS = [
    "perturb_source_path",
    "perturb_split",
    "perturb_work_dir",
    "perturb_candidate_path",
    "perturb_verified_path",
    "perturb_final_path",
    "perturb_num_shards",
    "perturb_shard_id",
    "perturb_build_workers",
    "perturb_verify_workers",
    "perturb_progress_every",
    "perturb_max_retries_same_backend",
]


def _load_cfg_from_json(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--config must be a JSON object")
    return data


def _maybe_set(kwargs: dict, key: str, value):
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    kwargs[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Semantic perturb CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common(sub):
        sub.add_argument("--config", type=str, required=True, help="Path to perturb_* JSON config")
        sub.add_argument("--perturb_source_path", type=str, default="")
        sub.add_argument("--perturb_split", type=str, default="")
        sub.add_argument("--perturb_work_dir", type=str, default="")
        sub.add_argument("--perturb_candidate_path", type=str, default="")
        sub.add_argument("--perturb_verified_path", type=str, default="")
        sub.add_argument("--perturb_final_path", type=str, default="")
        sub.add_argument("--perturb_num_shards", type=int, default=None)
        sub.add_argument("--perturb_shard_id", type=int, default=None)
        sub.add_argument("--perturb_build_workers", type=int, default=None)
        sub.add_argument("--perturb_verify_workers", type=int, default=None)
        sub.add_argument("--perturb_progress_every", type=int, default=None)
        sub.add_argument("--perturb_max_retries_same_backend", type=int, default=None)
        sub.add_argument("--perturb_resume", action="store_true", default=None)

    build = subparsers.add_parser("build-cache", help="Run generator-only candidate cache build")
    _add_common(build)

    verify = subparsers.add_parser("verify-cache", help="Run verifier stage on candidate cache")
    _add_common(verify)

    export = subparsers.add_parser("export-final", help="Export final cache from verified candidates")
    _add_common(export)

    retry = subparsers.add_parser("retry-failed", help="Retry only failed samples")
    _add_common(retry)
    retry.add_argument("--perturb_retry_stage", type=str, default="", choices=["build", "verify", "export"])
    retry.add_argument("--perturb_failure_path", type=str, default="")

    return parser


def _build_config(args) -> PerturbConfig:
    kwargs = _load_cfg_from_json(args.config)
    for key in COMMON_OVERRIDE_KEYS + ["perturb_retry_stage", "perturb_failure_path"]:
        _maybe_set(kwargs, key, getattr(args, key, None))
    _maybe_set(kwargs, "perturb_resume", getattr(args, "perturb_resume", None))

    return PerturbConfig(**kwargs)


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _build_config(args)

    if args.command == "build-cache":
        outputs = build_cache(cfg)
    elif args.command == "verify-cache":
        outputs = verify_cache(cfg)
    elif args.command == "export-final":
        outputs = export_final(cfg)
    elif args.command == "retry-failed":
        outputs = retry_failed(cfg)
    else:
        raise ValueError("Unsupported command '{}'".format(args.command))

    cfg_dump = asdict(cfg)
    if cfg_dump.get("perturb_llm_api_key"):
        cfg_dump["perturb_llm_api_key"] = "***REDACTED***"
    print(json.dumps({"command": args.command, "config": cfg_dump, "outputs": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
