#!/usr/bin/env bash
# Usage:
# bash method_tvr/scripts/run_exp1_gar.sh dset_name ctx_mode vid_feat_type exp_id [extra args]
set -euo pipefail

dset_name=$1
ctx_mode=$2
vid_feat_type=$3
exp_id=$4
shift 4

bash method_tvr/scripts/train.sh "${dset_name}" "${ctx_mode}" "${vid_feat_type}" \
  --exp_id "${exp_id}" \
  --backbone_type BiMamba \
  --use_generative_augmentation \
  --use_fusion_encoder \
  --lm_weight 0.3 \
  "$@"
