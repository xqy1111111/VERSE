#!/usr/bin/env bash
# Usage:
# bash method_tvr/scripts/run_exp2_tfvtg.sh model_dir split_name [dset_name] [extra args]
set -euo pipefail

model_dir=$1
split_name=$2
dset_name=${3:-""}
shift_args=2
if [[ ${dset_name} == "tvr" || ${dset_name} == "charades_fig" || ${dset_name} == "didemo_fig" || ${dset_name} == "activitynet_fig" ]]; then
    shift_args=3
else
    dset_name=""
fi
shift ${shift_args}

bash method_tvr/scripts/inference.sh "${model_dir}" "${split_name}" ${dset_name} \
  --scoring_method TFVTG \
  --tfvtg_dynamic_weight 0.5 \
  --tfvtg_static_weight 0.5 \
  --tfvtg_smooth_win 3 \
  "$@"
