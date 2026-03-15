#!/usr/bin/env bash
# Download VERIFIED *-FIG annotations.
# Usage:
# bash method_tvr/scripts/download_fig_annotations.sh [charades_fig didemo_fig activitynet_fig]
set -euo pipefail

datasets=("$@")
if [[ ${#datasets[@]} -eq 0 ]]; then
    datasets=("charades_fig" "didemo_fig" "activitynet_fig")
fi

base_url="https://raw.githubusercontent.com/hlchen23/VERIFIED/main/fine-grained-anno"
for dset in "${datasets[@]}"; do
    out_dir="data/fig/${dset}/annotations"
    mkdir -p "${out_dir}"
    case ${dset} in
        charades_fig)
            curl -L "${base_url}/charades-fig/charades_fig_train.jsonl" -o "${out_dir}/charades_fig_train.jsonl"
            curl -L "${base_url}/charades-fig/charades_fig_test.jsonl" -o "${out_dir}/charades_fig_test.jsonl"
            ;;
        didemo_fig)
            curl -L "${base_url}/didemo-fig/didemo_fig_train.jsonl" -o "${out_dir}/didemo_fig_train.jsonl"
            curl -L "${base_url}/didemo-fig/didemo_fig_val.jsonl" -o "${out_dir}/didemo_fig_val.jsonl"
            curl -L "${base_url}/didemo-fig/didemo_fig_test.jsonl" -o "${out_dir}/didemo_fig_test.jsonl"
            ;;
        activitynet_fig)
            curl -L "${base_url}/activitynet-fig/activitynet_fig_train.jsonl" -o "${out_dir}/activitynet_fig_train.jsonl"
            curl -L "${base_url}/activitynet-fig/activitynet_fig_val_1.jsonl" -o "${out_dir}/activitynet_fig_val_1.jsonl"
            curl -L "${base_url}/activitynet-fig/activitynet_fig_val_2.jsonl" -o "${out_dir}/activitynet_fig_val_2.jsonl"
            ;;
        *)
            echo "Unknown dataset ${dset}"
            exit 1
            ;;
    esac
done
