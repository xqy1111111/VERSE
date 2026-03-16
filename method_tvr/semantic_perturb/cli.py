import argparse
import json
from dataclasses import asdict

from method_tvr.semantic_perturb.cache_builder import SemanticBuildConfig, build_semantic_cache
from method_tvr.semantic_perturb.schema import NEG_TYPES, POS_TYPES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Semantic perturbation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-cache", help="Build offline Semantic cache with strict LLM generation/verifier")
    build.add_argument("--config", type=str, default="", help="Path to JSON config file")
    build.add_argument("--dset_name", type=str, default="")
    build.add_argument("--source_path", type=str, default="")
    build.add_argument("--cache_split", type=str, default="train")
    build.add_argument("--output_path", type=str, default="")
    build.add_argument("--backend", type=str, default="llm", choices=["llm"])
    build.add_argument("--strict_mode", action="store_true", default=None)
    build.add_argument("--no_strict_mode", action="store_true", default=None)
    build.add_argument("--no_fallback", action="store_true", default=None)
    build.add_argument("--allow_fallback", action="store_true", default=None)
    build.add_argument("--num_hard_neg", type=int, default=None)
    build.add_argument("--num_hard_pos", type=int, default=None)
    build.add_argument("--max_retries_same_backend", type=int, default=None)
    build.add_argument("--prompt_version", type=str, default="")
    build.add_argument("--schema_version", type=str, default="")
    build.add_argument("--generator_model", type=str, default="")
    build.add_argument("--verifier_model", type=str, default="")
    build.add_argument("--temperature", type=float, default=None)
    build.add_argument("--seed", type=int, default=None)
    build.add_argument("--neg_types", nargs="+", default=[])
    build.add_argument("--pos_types", nargs="+", default=[])
    build.add_argument("--severity_levels", nargs="+", type=int, default=[])
    build.add_argument("--llm_api_base", type=str, default="")
    build.add_argument("--llm_api_key", type=str, default="")
    build.add_argument("--llm_transport", type=str, default="", choices=["remote_api", "local_xgrammar"])
    build.add_argument("--llm_response_mode", type=str, default="", choices=["json_schema", "none"])
    build.add_argument("--local_model_name_or_path", type=str, default="")
    build.add_argument("--local_device", type=str, default="")
    build.add_argument("--local_mask_backend", type=str, default="")
    build.add_argument("--local_max_new_tokens", type=int, default=None)
    build.add_argument("--progress_every", type=int, default=None)
    build.add_argument("--slow_record_warn_s", type=float, default=None)
    build.add_argument("--build_workers", type=int, default=None)

    return parser


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
    if isinstance(value, list) and not value:
        return
    kwargs[key] = value


def _build_config_from_args(args) -> SemanticBuildConfig:
    from_file = _load_cfg_from_json(args.config)
    kwargs = dict(from_file)

    _maybe_set(kwargs, "dset_name", args.dset_name)
    _maybe_set(kwargs, "source_path", args.source_path)
    _maybe_set(kwargs, "cache_split", args.cache_split)
    _maybe_set(kwargs, "output_path", args.output_path)
    _maybe_set(kwargs, "backend", args.backend)
    _maybe_set(kwargs, "num_hard_neg", args.num_hard_neg)
    _maybe_set(kwargs, "num_hard_pos", args.num_hard_pos)
    _maybe_set(kwargs, "max_retries_same_backend", args.max_retries_same_backend)
    _maybe_set(kwargs, "prompt_version", args.prompt_version)
    _maybe_set(kwargs, "schema_version", args.schema_version)
    _maybe_set(kwargs, "generator_model", args.generator_model)
    _maybe_set(kwargs, "verifier_model", args.verifier_model)
    _maybe_set(kwargs, "temperature", args.temperature)
    _maybe_set(kwargs, "seed", args.seed)
    _maybe_set(kwargs, "llm_api_base", args.llm_api_base)
    _maybe_set(kwargs, "llm_api_key", args.llm_api_key)
    _maybe_set(kwargs, "llm_transport", args.llm_transport)
    _maybe_set(kwargs, "llm_response_mode", args.llm_response_mode)
    _maybe_set(kwargs, "local_model_name_or_path", args.local_model_name_or_path)
    _maybe_set(kwargs, "local_device", args.local_device)
    _maybe_set(kwargs, "local_mask_backend", args.local_mask_backend)
    _maybe_set(kwargs, "local_max_new_tokens", args.local_max_new_tokens)
    _maybe_set(kwargs, "progress_every", args.progress_every)
    _maybe_set(kwargs, "slow_record_warn_s", args.slow_record_warn_s)
    _maybe_set(kwargs, "build_workers", args.build_workers)

    if args.neg_types:
        kwargs["neg_types"] = list(args.neg_types)
    if args.pos_types:
        kwargs["pos_types"] = list(args.pos_types)
    if args.severity_levels:
        kwargs["severity_levels"] = list(args.severity_levels)

    if args.strict_mode:
        kwargs["strict_mode"] = True
    if args.no_strict_mode:
        kwargs["strict_mode"] = False
    if args.no_fallback:
        kwargs["no_fallback"] = True
    if args.allow_fallback:
        kwargs["no_fallback"] = False

    kwargs.setdefault("neg_types", sorted(NEG_TYPES))
    kwargs.setdefault("pos_types", sorted(POS_TYPES))

    return SemanticBuildConfig(**kwargs)


def main() -> None:
    args = _parser().parse_args()

    if args.command == "build-cache":
        cfg = _build_config_from_args(args)
        out_paths = build_semantic_cache(cfg)
        config_dump = asdict(cfg)
        if config_dump.get("llm_api_key"):
            config_dump["llm_api_key"] = "***REDACTED***"
        print("Semantic cache build complete")
        print(json.dumps({"config": config_dump, "outputs": out_paths}, indent=2, sort_keys=True))
        return

    raise ValueError("Unsupported command '{}'".format(args.command))


if __name__ == "__main__":
    main()
