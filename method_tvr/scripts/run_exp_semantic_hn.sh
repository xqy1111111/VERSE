#!/usr/bin/env bash
# Usage:
# bash method_tvr/scripts/run_exp_semantic_hn.sh dset_name ctx_mode vid_feat_type exp_id semantic_cache_path [extra args]
set -euo pipefail

dset_name=$1
ctx_mode=$2
vid_feat_type=$3
exp_id=$4
semantic_cache_path=$5
shift 5

bash method_tvr/scripts/train.sh "${dset_name}" "${ctx_mode}" "${vid_feat_type}" \
  --exp_id "${exp_id}" \
  --backbone_type BiMamba \
  --semantic_enable \
  --semantic_backend llm \
  --semantic_cache_path "${semantic_cache_path}" \
  --semantic_num_hard_neg 2 \
  --semantic_num_hard_pos 0 \
  --semantic_use_preference_loss \
  --semantic_preference_margin 0.2 \
  --semantic_preference_weight 1.0 \
  "$@"
