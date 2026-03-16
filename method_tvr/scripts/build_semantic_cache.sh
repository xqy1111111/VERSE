#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/setup.sh"
cd "${ROOT_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash method_tvr/scripts/build_semantic_cache.sh dset_name output_path [extra args]"
  exit 1
fi

dset_name=$1
output_path=$2
shift 2

VERIFIED_ROOT="${VERIFIED_ROOT:-/data/VERIFIED_FIG_2024}"
FIG_ANNO_ROOT="${FIG_ANNO_ROOT:-${VERIFIED_ROOT}/VERIFIED/fine-grained-anno}"

case ${dset_name} in
  tvr)
    train_path=data/tvr_train_release.jsonl
    ;;
  charades_fig)
    train_path=${FIG_ANNO_ROOT}/charades-fig/charades_fig_train.jsonl
    ;;
  didemo_fig)
    train_path=${FIG_ANNO_ROOT}/didemo-fig/didemo_fig_train.jsonl
    ;;
  activitynet_fig)
    train_path=${FIG_ANNO_ROOT}/activitynet-fig/activitynet_fig_train.jsonl
    ;;
  *)
    echo "Unknown dataset ${dset_name}"
    exit 1
    ;;
esac

api_base="${SEMANTIC_LLM_API_BASE:-${SILICONFLOW_API_BASE:-}}"
api_key="${SEMANTIC_LLM_API_KEY:-${SILICONFLOW_API_KEY:-}}"

cmd=(
  python -m method_tvr.semantic_perturb.cli build-cache
  --dset_name "${dset_name}"
  --source_path "${train_path}"
  --cache_split train
  --output_path "${output_path}"
  --backend llm
  --strict_mode
  --no_fallback
  --llm_transport "${SEMANTIC_LLM_TRANSPORT:-remote_api}"
  --llm_response_mode "${SEMANTIC_LLM_RESPONSE_MODE:-json_schema}"
)

if [[ -n "${SEMANTIC_LOCAL_MODEL_NAME_OR_PATH:-}" ]]; then
  cmd+=(--local_model_name_or_path "${SEMANTIC_LOCAL_MODEL_NAME_OR_PATH}")
fi
if [[ -n "${SEMANTIC_LOCAL_DEVICE:-}" ]]; then
  cmd+=(--local_device "${SEMANTIC_LOCAL_DEVICE}")
fi
if [[ -n "${SEMANTIC_LOCAL_MASK_BACKEND:-}" ]]; then
  cmd+=(--local_mask_backend "${SEMANTIC_LOCAL_MASK_BACKEND}")
fi
if [[ -n "${SEMANTIC_LOCAL_MAX_NEW_TOKENS:-}" ]]; then
  cmd+=(--local_max_new_tokens "${SEMANTIC_LOCAL_MAX_NEW_TOKENS}")
fi
if [[ -n "${api_base}" ]]; then
  cmd+=(--llm_api_base "${api_base}")
fi
if [[ -n "${api_key}" ]]; then
  cmd+=(--llm_api_key "${api_key}")
fi

cmd+=("$@")
"${cmd[@]}"
