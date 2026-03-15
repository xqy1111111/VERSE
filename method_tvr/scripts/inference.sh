#!/usr/bin/env bash
# run at project root dir
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/setup.sh"
cd "${ROOT_DIR}"
VERIFIED_ROOT="${VERIFIED_ROOT:-/home/qyxiao/data/VERIFIED_FIG_2024}"
FIG_ANNO_ROOT="${FIG_ANNO_ROOT:-${VERIFIED_ROOT}/VERIFIED/fine-grained-anno}"
# Usage:
# bash method/scripts/inference.sh MODEL_DIR SPLIT_NAME [dset_name] [ANY_OTHER_PYTHON_ARGS]
if [[ $# -lt 2 ]]; then
    echo "Usage: bash method_tvr/scripts/inference.sh MODEL_DIR SPLIT_NAME [dset_name] [extra args]"
    exit 1
fi
model_dir=$1
eval_split_name=$2  # [val | test | val_1 | val_2]
dset_name=${3:-""}
shift_args=2
if [[ ${dset_name} == "tvr" || ${dset_name} == "charades_fig" || ${dset_name} == "didemo_fig" || ${dset_name} == "activitynet_fig" ]]; then
    shift_args=3
else
    dset_name=""
fi
shift ${shift_args}

if [[ -z ${dset_name} || ${dset_name} == "tvr" ]]; then
    eval_path=data/tvr_${eval_split_name}_release.jsonl
else
    case ${dset_name} in
        charades_fig)
            eval_path=${FIG_ANNO_ROOT}/charades-fig/charades_fig_${eval_split_name}.jsonl
            ;;
        didemo_fig)
            eval_path=${FIG_ANNO_ROOT}/didemo-fig/didemo_fig_${eval_split_name}.jsonl
            ;;
        activitynet_fig)
            eval_path=${FIG_ANNO_ROOT}/activitynet-fig/activitynet_fig_${eval_split_name}.jsonl
            ;;
        *)
            echo "Unknown dataset ${dset_name}"
            exit 1
            ;;
    esac
fi
tasks=()
tasks+=(VCMR)
tasks+=(SVMR)
tasks+=(VR)
echo "tasks ${tasks[@]}"
python method_tvr/inference.py \
--model_dir ${model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
"$@"
