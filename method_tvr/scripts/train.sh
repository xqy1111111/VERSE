#!/usr/bin/env bash
# run at project root dir
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/setup.sh"
cd "${ROOT_DIR}"

if [[ $# -lt 3 ]]; then
    echo "Usage: bash method_tvr/scripts/train.sh dset_name ctx_mode vid_feat_type [extra args]"
    exit 1
fi
# Usage:
# bash method/scripts/train.sh tvr all ANY_OTHER_PYTHON_ARGS
# use --eval_tasks_at_training ["VR", "SVMR", "VCMR"] --stop_task ["VR", "SVMR", "VCMR"] for
# use --lw_neg_q 0 --lw_neg_ctx 0 for training SVMR/SVMR only
# use --lw_st_ed 0 for training with VR only
dset_name=$1  # see case below
ctx_mode=$2  # [video, tef, video_tef]
vid_feat_type=$3  # [resnet, i3d, resnet_i3d]
feature_root=data/tvr_feature_release
results_root=method_tvr/results
vid_feat_size=2048
extra_args=()
VERIFIED_ROOT="${VERIFIED_ROOT:-/home/qyxiao/data/VERIFIED_FIG_2024}"
FIG_ANNO_ROOT="${FIG_ANNO_ROOT:-${VERIFIED_ROOT}/VERIFIED/fine-grained-anno}"
FIG_FEAT_ROOT="${FIG_FEAT_ROOT:-${VERIFIED_ROOT}/features/VERIFIED_features/VERIFIED}"

if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
    echo "Subtitles are disabled in this project."
    exit 1
fi


case ${dset_name} in
    tvr)
        train_path=data/tvr_train_release.jsonl
        video_duration_idx_path=data/tvr_video2dur_idx.json
        desc_bert_path=${feature_root}/bert_feature/query_only/tvr_query_pretrained_w_query.h5
        if [[ ${vid_feat_type} == "i3d" ]]; then
            echo "Using I3D feature with shape 1024"
            vid_feat_path=${feature_root}/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5
            vid_feat_size=1024
        elif [[ ${vid_feat_type} == "resnet" ]]; then
            echo "Using ResNet feature with shape 2048"
            vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5
            vid_feat_size=2048
        elif [[ ${vid_feat_type} == "resnet_i3d" ]]; then
            echo "Using concatenated ResNet and I3D feature with shape 2048+1024"
            vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5
            vid_feat_size=3072
            extra_args+=(--no_norm_vfeat)  # since they are already normalized.
        fi
        eval_split_name=val
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=(data/tvr_val_release.jsonl)
        clip_length=1.5
        # extra_args+=(--max_ctx_l)
        # extra_args+=(100)  # max_ctx_l = 100 for clip_length = 1.5, only ~109/21825 has more than 100.
        extra_args+=(--max_pred_l)
        extra_args+=(16)
        if [[ ${vid_feat_type} != "i3d" && ${vid_feat_type} != "resnet" && ${vid_feat_type} != "resnet_i3d" ]]; then
            echo "Unknown vid_feat_type ${vid_feat_type}"
            exit 1
        fi
        ;;
    charades_fig)
        train_path=${FIG_ANNO_ROOT}/charades-fig/charades_fig_train.jsonl
        eval_split_name=test
        eval_path=${FIG_ANNO_ROOT}/charades-fig/charades_fig_test.jsonl
        video_duration_idx_path=${FIG_FEAT_ROOT}/Charades-FIG/video_feature/cha_video2dur_idx.json
        desc_bert_path=${FIG_FEAT_ROOT}/Charades-FIG/new_desc_feature/vcmr_roberta_base_cha_embed.h5
        if [[ ${vid_feat_type} != "resnet" ]]; then
            echo "Charades-FIG only supports resnet features."
            exit 1
        fi
        vid_feat_path=${FIG_FEAT_ROOT}/Charades-FIG/video_feature/charades_resnet152_4fps_max_1fps.h5
        vid_feat_size=2048
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=(${eval_path})
        clip_length=1.0
        ;;
    didemo_fig)
        train_path=${FIG_ANNO_ROOT}/didemo-fig/didemo_fig_train.jsonl
        eval_split_name=val
        eval_path=${FIG_ANNO_ROOT}/didemo-fig/didemo_fig_val.jsonl
        video_duration_idx_path=${FIG_FEAT_ROOT}/DiDeMo-FIG/video_feature/didemo_video2dur_idx_filter_unexist.json
        desc_bert_path=${FIG_FEAT_ROOT}/DiDeMo-FIG/new_desc_feature/vcmr_roberta_base_didemo_embed.h5
        if [[ ${vid_feat_type} != "resnet" ]]; then
            echo "DiDeMo-FIG only supports resnet features."
            exit 1
        fi
        vid_feat_path=${FIG_FEAT_ROOT}/DiDeMo-FIG/video_feature/didemo_resnet152_4fps_max_1fps.h5
        vid_feat_size=2048
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=(${eval_path})
        clip_length=1.0
        ;;
    activitynet_fig)
        train_path=${FIG_ANNO_ROOT}/activitynet-fig/activitynet_fig_train.jsonl
        eval_split_name=val_1
        eval_path=${FIG_ANNO_ROOT}/activitynet-fig/activitynet_fig_val_1.jsonl
        video_duration_idx_path=${FIG_FEAT_ROOT}/ActivityNet-FIG/video_feature/anet_video2dur_idx_filter_unexist.json
        desc_bert_path=${FIG_FEAT_ROOT}/ActivityNet-FIG/new_desc_feature/vcmr_roberta_base_anet_embed.h5
        if [[ ${vid_feat_type} != "resnet" ]]; then
            echo "ActivityNet-FIG only supports resnet features."
            exit 1
        fi
        vid_feat_path=${ANET_FIG_VID_FEAT_PATH:-${FIG_FEAT_ROOT}/ActivityNet-FIG/video_feature/activitynet_fig_resnet152.h5}
        if [[ ! -f ${vid_feat_path} ]]; then
            echo "ActivityNet-FIG requires a merged single H5 feature file."
            echo "Set ANET_FIG_VID_FEAT_PATH to the merged file path before training."
            exit 1
        fi
        vid_feat_size=2048
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=(${eval_path})
        clip_length=1.0
        ;;
    *)
        echo "Unknown dataset ${dset_name}"
        exit 1
        ;;
esac

echo "Start training with dataset [${dset_name}] in Context Mode [${ctx_mode}]"
echo "Extra args ${extra_args[@]}"
python method_tvr/train.py \
--dset_name=${dset_name} \
--eval_split_name=${eval_split_name} \
--nms_thd=${nms_thd} \
--results_root=${results_root} \
--train_path=${train_path} \
--desc_bert_path=${desc_bert_path} \
--video_duration_idx_path=${video_duration_idx_path} \
--vid_feat_path=${vid_feat_path} \
--clip_length=${clip_length} \
--vid_feat_size=${vid_feat_size} \
--ctx_mode=${ctx_mode} \
${extra_args[@]} \
${@:4}
