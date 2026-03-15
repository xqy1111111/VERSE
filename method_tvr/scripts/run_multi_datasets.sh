#!/usr/bin/env bash
# Usage:
# bash method_tvr/scripts/run_multi_datasets.sh "charades_fig didemo_fig" video_tef resnet exp_prefix [extra args]
set -euo pipefail

datasets_str=$1
ctx_mode=$2
vid_feat_type=$3
exp_prefix=$4
shift 4

IFS=' ' read -r -a datasets <<< "${datasets_str}"
for dset in "${datasets[@]}"; do
    bash method_tvr/scripts/train.sh "${dset}" "${ctx_mode}" "${vid_feat_type}" \
        --exp_id "${exp_prefix}_${dset}" \
        "$@"
done
