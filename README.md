# VERSE: Verified Enrichment with Re-ranking and Structured Estimation

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-under%20review-yellow)

---

## Overview

VERSE is a Video Corpus Moment Retrieval (VCMR) system that jointly addresses video retrieval, temporal localization, and fine-grained ranking in a single unified pipeline. It introduces three tightly integrated contributions: a Build-Verify-Export (BVE) pipeline that uses LLM-driven generation and verification to enrich training data with semantically precise hard positives and negatives; a token-aware late interaction module that performs multi-vector MaxSim reranking over top-K candidate videos; and a legality-aware biaffine span head that models start and end positions jointly via a 2D scoring matrix. VERSE is evaluated on the VERIFIED benchmark, specifically the Charades-FIG and DiDeMo-FIG fine-grained annotation splits.

VERSE extends the dual-encoder retrieval-localization framework of ReLoCLNet with three independently ablatable modules:

**BVE pipeline.** A three-stage offline pipeline (Build, Verify, Export) uses a large language model to generate candidate hard negatives (attribute swaps, action swaps, role swaps, temporal flips, etc.) and hard positives (paraphrases, syntax reorders, lexical variants) for each training query. A separate LLM verifier accepts or rejects each candidate, and only verified entries enter the frozen training cache. At training time, compositional supervision applies positive invariance and negative preference losses with a configurable warmup and ramp schedule.

**Token-aware late interaction.** After an initial single-vector retrieval pass that scores all corpus videos efficiently, the system selects up to `K` content-bearing query token vectors (with optional local phrase pooling) and computes ColBERT-style MaxSim scores over the top-100 candidate videos. The resulting late score is fused as a residual term with rank-aware attenuation and a hard-query gate that protects the baseline VR ranking from over-aggressive updates.

**Biaffine span head.** The standard independent 1D start/end prediction head is replaced by a biaffine span head that scores all valid (start, end) pairs jointly under an upper-triangular legality mask, enabling the model to capture span-level dependencies that two independent classifiers cannot.

The architecture diagram is available in [figures/method.png](figures/method.png).

---

## Requirements

VERSE requires Python 3.9-3.11 and a CUDA-capable GPU. Core dependencies:

| Package | Version |
|---|---|
| torch | 2.2.2 (cu118) |
| torchvision | 0.17.2 |
| transformers | >=4.38.2, <5 |
| tokenizers | >=0.15.2, <0.20 |
| h5py | >=3.10, <4 |
| numpy | >=1.24, <2 |
| tensorboard | >=2.15, <3 |
| easydict | >=1.13, <2 |
| tqdm | >=4.66, <5 |
| mamba-ssm | 2.3.0 (for BiMamba backbone) |

The full pinned dependency set is in [uv.lock](uv.lock).

---

## Installation

The recommended path uses [uv](https://github.com/astral-sh/uv) for reproducible environment management.

```bash
git clone https://github.com/xqy1111111/VERSE.git
cd VERSE

# Install uv if not already present
pip install uv

# Create environment with Python 3.9 and install all dependencies
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate
uv sync --extra gpu-cu118 --extra bimamba

# Add project root to PYTHONPATH (required every new shell session)
source setup.sh
```

If you prefer conda, a compatible environment can be built manually from the package versions listed in [pyproject.toml](pyproject.toml).

---

## Data Preparation

VERSE is evaluated on the VERIFIED benchmark. You need two things: the fine-grained annotations (JSONL) and the pre-extracted video/text features (H5).

### Step 1: Download annotations

The VERIFIED annotation files are publicly available from the [VERIFIED repository](https://github.com/hlchen23/VERIFIED).

```bash
# Download Charades-FIG and DiDeMo-FIG annotations to data/fig/
bash method_tvr/scripts/download_fig_annotations.sh charades_fig didemo_fig
```

This creates:

```
data/fig/
  charades_fig/
    annotations/
      charades_fig_train.jsonl
      charades_fig_test.jsonl
  didemo_fig/
    annotations/
      didemo_fig_train.jsonl
      didemo_fig_val.jsonl
      didemo_fig_test.jsonl
```

### Step 2: Download video features and pre-extracted text features

Pre-extracted ResNet-152 video features and RoBERTa-base query features are distributed with the VERIFIED benchmark. Follow the download instructions in the [VERIFIED repository](https://github.com/hlchen23/VERIFIED) to obtain:

```
# TODO: set VERIFIED_ROOT to the directory where you extracted the VERIFIED feature package
export VERIFIED_ROOT=/path/to/VERIFIED_FIG_2024
export FIG_FEAT_ROOT=${VERIFIED_ROOT}/features/VERIFIED_features/VERIFIED
export FIG_ANNO_ROOT=${VERIFIED_ROOT}/VERIFIED/fine-grained-anno
```

The training script resolves feature paths from these environment variables automatically.

### Step 3: Build the video duration index

```bash
python method_tvr/scripts/build_video2dur_idx.py \
  --input_jsonl ${FIG_ANNO_ROOT}/charades-fig/charades_fig_train.jsonl \
  --output_json ${FIG_FEAT_ROOT}/Charades-FIG/video_feature/cha_video2dur_idx.json
```

### Step 4 (optional): Re-extract query features

If you need to re-extract RoBERTa-base query features from the raw annotation JSONL:

```bash
python method_tvr/scripts/build_desc_bert_h5.py \
  --input_jsonl ${FIG_ANNO_ROOT}/charades-fig/charades_fig_train.jsonl \
  --output_h5 ${FIG_FEAT_ROOT}/Charades-FIG/new_desc_feature/vcmr_roberta_base_cha_embed.h5 \
  --tokenizer roberta-base \
  --max_len 30
```

### Step 5 (optional): Build the BVE semantic cache

The BVE training cache is pre-built offline and committed to `cache/`. If you want to rebuild it from scratch using your own LLM API:

```bash
python method_tvr/train.py \
  --dset_name charades_fig \
  --ctx_mode video_tef \
  --exp_id semantic_cache_build_charades \
  --train_path ${FIG_ANNO_ROOT}/charades-fig/charades_fig_train.jsonl \
  --eval_path  ${FIG_ANNO_ROOT}/charades-fig/charades_fig_test.jsonl \
  --desc_bert_path ${FIG_FEAT_ROOT}/Charades-FIG/new_desc_feature/vcmr_roberta_base_cha_embed.h5 \
  --video_duration_idx_path ${FIG_FEAT_ROOT}/Charades-FIG/video_feature/cha_video2dur_idx.json \
  --vid_feat_path ${FIG_FEAT_ROOT}/Charades-FIG/video_feature/charades_resnet152_4fps_max_1fps.h5 \
  --vid_feat_size 2048 \
  --semantic_enable --semantic_backend llm \
  --semantic_build_cache_only \
  --semantic_cache_path cache/charades_fig/train/semantic_perturb_train.jsonl \
  --semantic_no_fallback --semantic_strict_mode \
  --semantic_fail_on_missing_cache --semantic_fail_on_invalid_cache \
  --semantic_num_hard_neg 2 --semantic_num_hard_pos 2 \
  --semantic_llm_transport remote_api \
  --semantic_llm_response_mode json_schema \
  --semantic_generator_model deepseek-ai/DeepSeek-V3.2 \
  --semantic_verifier_model deepseek-ai/DeepSeek-V3.2
  # TODO: set --semantic_llm_api_key and --semantic_llm_api_base for your LLM provider
```

Alternatively, use the three-stage offline CLI:

```bash
python -m method_tvr.semantic_perturb.cli build-cache  --config /path/to/perturb_config.json
python -m method_tvr.semantic_perturb.cli verify-cache --config /path/to/perturb_config.json
python -m method_tvr.semantic_perturb.cli export-final --config /path/to/perturb_config.json
```

---

## Training

Set environment variables first:

```bash
export VERIFIED_ROOT=/path/to/VERIFIED_FIG_2024
export FIG_ANNO_ROOT=${VERIFIED_ROOT}/VERIFIED/fine-grained-anno
export FIG_FEAT_ROOT=${VERIFIED_ROOT}/features/VERIFIED_features/VERIFIED
source setup.sh
```

**Full VERSE model on Charades-FIG** (BiMamba backbone + BVE + late interaction + biaffine span head):

```bash
CUDA_VISIBLE_DEVICES=0 python -m method_tvr.train \
  --dset_name charades_fig \
  --eval_split_name test \
  --ctx_mode video_tef \
  --exp_id verse_charades \
  --train_path ${FIG_ANNO_ROOT}/charades-fig/charades_fig_train.jsonl \
  --eval_path  ${FIG_ANNO_ROOT}/charades-fig/charades_fig_test.jsonl \
  --desc_bert_path ${FIG_FEAT_ROOT}/Charades-FIG/new_desc_feature/vcmr_roberta_base_cha_embed.h5 \
  --video_duration_idx_path ${FIG_FEAT_ROOT}/Charades-FIG/video_feature/cha_video2dur_idx.json \
  --vid_feat_path ${FIG_FEAT_ROOT}/Charades-FIG/video_feature/charades_resnet152_4fps_max_1fps.h5 \
  --q_feat_size 768 --vid_feat_size 2048 --clip_length 1.0 \
  --seed 2018 --lr 8e-5 --lr_warmup_proportion 0.03 --wd 0.01 \
  --n_epoch 80 --max_es_cnt 18 --bsz 96 \
  --backbone_type BiMamba --hidden_size 1024 --n_heads 16 \
  --mamba_d_state 128 --mamba_d_conv 4 --mamba_expand 2 \
  --use_generative_augmentation --use_fusion_encoder \
  --fusion_num_layers 4 --lm_weight 0.3 --lm_max_len 30 --lm_num_layers 2 \
  --tokenizer_name_or_path roberta-base \
  --retrieval_scorer residual_rerank \
  --late_interaction_dim 384 --late_interaction_rerank_topk 100 \
  --late_interaction_train_rerank_topk 8 --late_interaction_eval_rerank_topk 100 \
  --late_interaction_score_weight 0.2 \
  --late_interaction_train_score_weight 0.02 --late_interaction_eval_score_weight 0.023 \
  --late_interaction_residual_clip 0.5 \
  --late_interaction_rank_head_weight 0.45 --late_interaction_rank_gamma 2.5 \
  --late_interaction_detach_backbone_in_train \
  --multi_vector_query_max_count 4 --multi_vector_phrase_window 2 \
  --span_head_type biaffine_span_head \
  --span_biaffine_hidden_size 192 --lw_span_joint 0.03 \
  --lw_st_ed 0.05 --lw_fcl 0.06 --lw_vcl 0.06 \
  --hard_negative_start_epoch 6 --hard_pool_size 40 \
  --semantic_enable --semantic_backend llm \
  --semantic_cache_path cache/charades_fig/train/semantic_perturb_train.jsonl \
  --semantic_strict_mode --semantic_no_fallback \
  --semantic_fail_on_missing_cache --semantic_fail_on_invalid_cache \
  --enable_compositional_supervision --require_rewrite_cache \
  --positive_rewrite_sample_size 2 --negative_rewrite_sample_size 2 \
  --positive_invariance_weight 1.0 --negative_preference_weight 1.0 \
  --enable_debiased_retrieval_correction --debiased_retrieval_weight 0.05 \
  --compositional_warmup_epochs 3 --compositional_ramp_epochs 6
```

**Full VERSE model on DiDeMo-FIG:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m method_tvr.train \
  --dset_name didemo_fig \
  --eval_split_name val \
  --ctx_mode video_tef \
  --exp_id verse_didemo \
  --train_path ${FIG_ANNO_ROOT}/didemo-fig/didemo_fig_train.jsonl \
  --eval_path  ${FIG_ANNO_ROOT}/didemo-fig/didemo_fig_val.jsonl \
  --desc_bert_path ${FIG_FEAT_ROOT}/DiDeMo-FIG/new_desc_feature/vcmr_roberta_base_didemo_embed.h5 \
  --video_duration_idx_path ${FIG_FEAT_ROOT}/DiDeMo-FIG/video_feature/didemo_video2dur_idx_filter_unexist.json \
  --vid_feat_path ${FIG_FEAT_ROOT}/DiDeMo-FIG/video_feature/didemo_resnet152_4fps_max_1fps.h5 \
  --q_feat_size 768 --vid_feat_size 2048 --clip_length 1.0 \
  --seed 2018 --lr 8e-5 --lr_warmup_proportion 0.03 --wd 0.01 \
  --n_epoch 80 --max_es_cnt 18 --bsz 96 \
  --backbone_type BiMamba --hidden_size 1024 --n_heads 16 \
  --mamba_d_state 128 --mamba_d_conv 4 --mamba_expand 2 \
  --use_generative_augmentation --use_fusion_encoder \
  --fusion_num_layers 4 --lm_weight 0.3 --lm_max_len 30 --lm_num_layers 2 \
  --tokenizer_name_or_path roberta-base \
  --retrieval_scorer residual_rerank \
  --late_interaction_dim 384 --late_interaction_rerank_topk 100 \
  --late_interaction_train_rerank_topk 8 --late_interaction_eval_rerank_topk 100 \
  --late_interaction_score_weight 0.2 \
  --late_interaction_train_score_weight 0.02 --late_interaction_eval_score_weight 0.023 \
  --late_interaction_residual_clip 0.5 \
  --late_interaction_rank_head_weight 0.45 --late_interaction_rank_gamma 2.5 \
  --late_interaction_detach_backbone_in_train \
  --multi_vector_query_max_count 4 --multi_vector_phrase_window 2 \
  --span_head_type biaffine_span_head \
  --span_biaffine_hidden_size 192 --lw_span_joint 0.03 \
  --lw_st_ed 0.05 --lw_fcl 0.06 --lw_vcl 0.06 \
  --hard_negative_start_epoch 6 --hard_pool_size 40 \
  --semantic_enable --semantic_backend llm \
  --semantic_cache_path cache/didemo_fig/train/semantic_perturb_train.jsonl \
  --semantic_strict_mode --semantic_no_fallback \
  --semantic_fail_on_missing_cache --semantic_fail_on_invalid_cache \
  --enable_compositional_supervision --require_rewrite_cache \
  --positive_rewrite_sample_size 2 --negative_rewrite_sample_size 2 \
  --positive_invariance_weight 1.0 --negative_preference_weight 1.0 \
  --enable_debiased_retrieval_correction --debiased_retrieval_weight 0.05 \
  --compositional_warmup_epochs 3 --compositional_ramp_epochs 6
```

Trained model checkpoints and evaluation logs are saved under `method_tvr/results/<dset>-<ctx_mode>-<exp_id>-<timestamp>/`.

---

## Evaluation

To evaluate a trained checkpoint on a held-out split:

```bash
bash method_tvr/scripts/inference.sh \
  method_tvr/results/<run_dir> \
  test \
  charades_fig
```

For DiDeMo-FIG validation:

```bash
bash method_tvr/scripts/inference.sh \
  method_tvr/results/<run_dir> \
  val \
  didemo_fig
```

Metrics are computed at IoU thresholds 0.5 and 0.7 for VCMR and SVMR, and without an IoU threshold for VR. The standalone evaluator in `standalone_eval/eval.py` can also be used directly on saved prediction files.

### Results on VERIFIED benchmark

Results reported below are for the full VERSE model (BiMamba + BVE + late interaction + biaffine span head).

**Charades-FIG** (test split):

| Task | IoU | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|---|
| VCMR | 0.5 | 1.37 | 3.12 | 4.27 | 11.59 |
| VCMR | 0.7 | 0.81 | 1.94 | 2.63 | 7.96 |
| SVMR | 0.5 | 35.40 | -- | -- | 88.82 |
| SVMR | 0.7 | 16.69 | -- | -- | 75.94 |
| VR   | -- | 2.55 | 8.39 | 12.90 | 49.46 |

**DiDeMo-FIG** (val split):

| Task | IoU | R@1 | R@5 | R@10 | R@100 |
|---|---|---|---|---|---|
| VCMR | 0.5 | 4.79 | 13.10 | 19.74 | 51.64 |
| VCMR | 0.7 | 2.61 | 8.62 | 14.08 | 44.93 |
| SVMR | 0.5 | 33.94 | -- | -- | 98.47 |
| SVMR | 0.7 | 19.83 | -- | -- | 96.67 |
| VR   | -- | 14.35 | 37.60 | 50.80 | 87.90 |

---

## Project Structure

```
VERSE/
├── method_tvr/                  # Core model and training code
│   ├── train.py                 # Training entry point
│   ├── inference.py             # Inference and evaluation entry point
│   ├── config.py                # All CLI arguments and option parsing
│   ├── model.py                 # Main ReLoCLNet model
│   ├── bimamba.py               # Bidirectional Mamba encoder layer
│   ├── late_interaction.py      # Token-aware late interaction reranker
│   ├── contrastive.py           # Video-query contrastive loss
│   ├── debiased_video_frame_loss.py  # Background-aware frame contrastive loss
│   ├── model_components.py      # Attention, positional encoding, MIL-NCE
│   ├── query_decoder.py         # Autoregressive query decoder for GAR loss
│   ├── proposal.py              # Proposal config and clip-length lookup
│   ├── optimization.py          # Optimizer and LR scheduler
│   ├── start_end_dataset.py     # Dataset class for retrieval-localization
│   ├── span_heads/
│   │   └── biaffine_span_head.py  # Legality-aware biaffine span head
│   ├── semantic_perturb/        # BVE pipeline (build, verify, export, train)
│   │   ├── cli.py               # Offline three-stage CLI entry point
│   │   ├── cache_builder.py     # One-shot training-integrated cache builder
│   │   ├── builder.py           # Staged offline build/verify/export logic
│   │   ├── dataset_semantic.py  # Cache loading and strict manifest validation
│   │   ├── generator.py         # LLM-based candidate generation
│   │   ├── verifier.py          # LLM-based candidate verification
│   │   ├── losses.py            # Compositional supervision losses
│   │   └── schema.py            # Cache entry schema and validation
│   └── scripts/
│       ├── train.sh             # Convenience training wrapper
│       ├── inference.sh         # Convenience inference wrapper
│       ├── download_fig_annotations.sh  # Download VERIFIED annotations
│       ├── build_desc_bert_h5.py        # Re-extract RoBERTa query features
│       └── build_video2dur_idx.py       # Build video duration index
├── standalone_eval/
│   └── eval.py                  # Standalone metric computation
├── utils/                       # General utilities (JSON, zip, logging)
├── data/
│   └── fig/                     # VERIFIED benchmark annotation files
├── cache/
│   ├── charades_fig/train/      # Pre-built BVE cache for Charades-FIG
│   └── didemo_fig/train/        # Pre-built BVE cache for DiDeMo-FIG
├── figures/
│   └── method.png               # Architecture overview diagram
├── setup.sh                     # Sets PYTHONPATH for the project root
├── pyproject.toml               # Dependency declaration (uv)
└── uv.lock                      # Pinned dependency lock file
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

VERSE builds on [ReLoCLNet](https://github.com/Tangshitao/ReLoCLNet) (SIGIR 2021, Zhang et al.) and [TVRetrieval](https://github.com/jayleicn/TVRetrieval).
