import os
import time
import torch
import argparse
from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile
from method_tvr.proposal import ProposalConfigs


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--dset_name", type=str,
                                 choices=["tvr", "charades_fig", "didemo_fig", "activitynet_fig"])
        self.parser.add_argument("--eval_split_name", type=str, default="val",
                                 help="should match keys in video_duration_idx_path, must set for VCMR")
        self.parser.add_argument("--debug", action="store_true",
                                 help="debug (fast) mode, break all loops, do not load all data into memory.")
        self.parser.add_argument("--data_ratio", type=float, default=1.0,
                                 help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                      "Use small portion for debug purposes. Note this is different from --debug, "
                                      "which works by breaking the loops, typically they are not used together.")
        self.parser.add_argument("--results_root", type=str, default="results")
        self.parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=8,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--no_pin_memory", action="store_true", help="No use pin_memory=True for dataloader")

        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_warmup_proportion", type=float, default=0.01,
                                 help="Proportion of training to perform linear learning rate warmup.")
        self.parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=10,
                                 help="number of epochs to early stop, use -1 to disable early stop")
        self.parser.add_argument("--stop_task", type=str, default="VCMR", choices=["VCMR", "SVMR", "VR"],
                                 help="Use metric associated with stop_task for early stop")
        self.parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                                 help="Minimum stop-score improvement to reset early-stop counter.")
        self.parser.add_argument("--early_stop_use_composite", action="store_true",
                                 help="Use weighted multi-task score for early stop instead of single stop_task.")
        self.parser.add_argument("--early_stop_vcmr_weight", type=float, default=1.0,
                                 help="Composite early-stop weight for VCMR strict score.")
        self.parser.add_argument("--early_stop_svmr_weight", type=float, default=1.0,
                                 help="Composite early-stop weight for SVMR strict score.")
        self.parser.add_argument("--early_stop_vr_weight", type=float, default=0.5,
                                 help="Composite early-stop weight for VR-r1 score.")
        self.parser.add_argument("--early_stop_normalize_components", action="store_true",
                                 help=("Normalize VCMR/SVMR/VR composite components before weighting. "
                                       "Useful because task score ranges are different."))
        self.parser.add_argument("--early_stop_vcmr_scale", type=float, default=1.0,
                                 help="Scale divisor for VCMR component when --early_stop_normalize_components is set.")
        self.parser.add_argument("--early_stop_svmr_scale", type=float, default=40.0,
                                 help="Scale divisor for SVMR component when --early_stop_normalize_components is set.")
        self.parser.add_argument("--early_stop_vr_scale", type=float, default=2.0,
                                 help="Scale divisor for VR component when --early_stop_normalize_components is set.")
        self.parser.add_argument("--early_stop_vcmr_metrics", type=str, nargs="+",
                                 default=["0.5-r1", "0.7-r1"],
                                 choices=[
                                     "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
                                     "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100"
                                 ],
                                 help=("VCMR metrics used in early-stop aggregation. "
                                       "Defaults to strict recall (0.5-r1 + 0.7-r1)."))
        self.parser.add_argument("--early_stop_svmr_metrics", type=str, nargs="+",
                                 default=["0.5-r1", "0.7-r1"],
                                 choices=[
                                     "0.5-r1", "0.5-r5", "0.5-r10", "0.5-r100",
                                     "0.7-r1", "0.7-r5", "0.7-r10", "0.7-r100"
                                 ],
                                 help=("SVMR metrics used in early-stop aggregation. "
                                       "Defaults to strict recall (0.5-r1 + 0.7-r1)."))
        self.parser.add_argument("--early_stop_vr_metrics", type=str, nargs="+",
                                 default=["r1"],
                                 choices=["r1", "r5", "r10", "r100"],
                                 help="VR metrics used in early-stop aggregation. Defaults to r1.")
        self.parser.add_argument("--eval_tasks_at_training", type=str, nargs="+", default=["VCMR", "SVMR", "VR"],
                                 choices=["VCMR", "SVMR", "VR"], help="evaluate and report numbers for tasks.")
        self.parser.add_argument("--bsz", type=int, default=128, help="mini-batch size")
        self.parser.add_argument("--eval_query_bsz", type=int, default=50, help="minibatch size at inference for query")
        self.parser.add_argument("--eval_context_bsz", type=int, default=200,
                                 help="mini-batch size at inference for context videos")
        self.parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        self.parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        self.parser.add_argument("--margin", type=float, default=0.1, help="margin for hinge loss")
        self.parser.add_argument("--lw_neg_q", type=float, default=1,
                                 help="weight for ranking loss with negative query and positive context")
        self.parser.add_argument("--lw_neg_ctx", type=float, default=1,
                                 help="weight for ranking loss with positive query and negative context")
        self.parser.add_argument("--lw_st_ed", type=float, default=0.01, help="weight for st ed prediction loss")
        self.parser.add_argument("--lw_fcl", type=float, default=0.03, help="weight for frame CL loss")
        self.parser.add_argument("--lw_vcl", type=float, default=0.03, help="weight for video CL loss")
        self.parser.add_argument("--train_span_start_epoch", type=int, default=0,
                                 help="which epoch to start training span prediction, -1 to disable")
        self.parser.add_argument("--ranking_loss_type", type=str, default="hinge", choices=["hinge", "lse"],
                                 help="att loss type, can be hinge loss or its smooth approximation LogSumExp")
        self.parser.add_argument("--hard_negative_start_epoch", type=int, default=20,
                                 help="which epoch to start hard negative sampling for video-level ranking loss,"
                                      "use -1 to disable")
        self.parser.add_argument("--hard_pool_size", type=int, default=20,
                                 help="hard negatives are still sampled, but from a harder pool.")

        self.parser.add_argument("--max_desc_l", type=int, default=30, help="max length of descriptions")
        self.parser.add_argument("--max_ctx_l", type=int, default=128,
                                 help="max number of snippets, 100 for tvr clip_length=1.5, oly 109/21825 > 100")
        self.parser.add_argument("--train_path", type=str, default=None)
        self.parser.add_argument("--eval_path", type=str, default=None,
                                 help="Evaluating during training, for Dev set. If None, will only do training, "
                                      "anet_cap and charades_sta has no dev set, so None")
        self.parser.add_argument("--desc_bert_path", type=str, default=None)
        self.parser.add_argument("--q_feat_size", type=int, default=768, help="feature dim for query feature")
        self.parser.add_argument("--ctx_mode", type=str, help="which context to use a combination of [video, tef]",
                                 choices=["video", "tef", "video_tef"])
        self.parser.add_argument("--video_duration_idx_path", type=str, default=None)
        self.parser.add_argument("--vid_feat_path", type=str, default="")
        self.parser.add_argument("--no_norm_vfeat", action="store_true",
                                 help="Do not do normalization on video feat, use it only when using resnet_i3d feat")
        self.parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalization on text feat")
        self.parser.add_argument("--clip_length", type=float, default=None,
                                 help="each video will be uniformly segmented into small clips, "
                                      "will automatically loaded from ProposalConfigs if None")
        self.parser.add_argument("--vid_feat_size", type=int, help="feature dim for video feature")
        self.parser.add_argument("--max_position_embeddings", type=int, default=300)
        self.parser.add_argument("--hidden_size", type=int, default=384)
        self.parser.add_argument("--n_heads", type=int, default=8)
        self.parser.add_argument("--input_drop", type=float, default=0.1, help="Applied to all inputs")
        self.parser.add_argument("--drop", type=float, default=0.1, help="Applied to all other layers")
        self.parser.add_argument("--conv_kernel_size", type=int, default=5)
        self.parser.add_argument("--conv_stride", type=int, default=1)
        self.parser.add_argument("--initializer_range", type=float, default=0.02, help="initializer range for layers")

        self.parser.add_argument("--backbone_type", type=str, default="Transformer",
                                 choices=["Transformer", "BiMamba"])
        self.parser.add_argument("--retrieval_scorer", type=str, default="single_vector",
                                 choices=["single_vector", "residual_rerank", "late_interaction", "combined"],
                                 help=("Video retrieval scorer: single_vector (baseline) or residual_rerank "
                                       "(baseline + late-interaction top-k refinement). "
                                       "late_interaction/combined are legacy aliases of residual_rerank."))
        self.parser.add_argument("--late_interaction_dim", type=int, default=0,
                                 help="Late interaction vector dim. <=0 means using hidden_size.")
        self.parser.add_argument("--late_interaction_no_projection", action="store_true",
                                 help="Disable projection layers for late interaction retrieval.")
        self.parser.add_argument("--late_interaction_use_token_weight", action="store_true",
                                 help="Enable learned query token weighting in late interaction retrieval.")
        self.parser.add_argument("--late_interaction_token_weight_floor", type=float, default=0.0,
                                 help="Suppress low-weight query tokens below this threshold.")
        self.parser.add_argument("--late_interaction_score_reduction", type=str, default="mean",
                                 choices=["sum", "mean"],
                                 help="Reduce token-level MaxSim by sum or mean.")
        self.parser.add_argument("--late_interaction_video_chunk_size", type=int, default=256,
                                 help="Chunk size over videos for late interaction scoring.")
        self.parser.add_argument("--late_interaction_rerank_topk", type=int, default=50,
                                 help="Top-k candidates from baseline retrieval to refine with late interaction.")
        self.parser.add_argument("--late_interaction_train_rerank_topk", type=int, default=-1,
                                 help=("Train-time top-k for late rerank. "
                                       "<0 means using --late_interaction_rerank_topk."))
        self.parser.add_argument("--late_interaction_eval_rerank_topk", type=int, default=-1,
                                 help=("Eval-time top-k for late rerank. "
                                       "<0 means using --late_interaction_rerank_topk."))
        self.parser.add_argument("--late_interaction_query_chunk_size", type=int, default=4,
                                 help="Query chunk size for batched late-interaction rerank scoring.")
        self.parser.add_argument("--late_interaction_rerank_margin_threshold", type=float, default=-1.0,
                                 help=("Only rerank hard queries with (top1-top2)<=threshold. "
                                       "<0 disables hard-query gating."))
        self.parser.add_argument("--late_interaction_rerank_soft_temperature", type=float, default=0.0,
                                 help=("Enable soft rerank gating with sigmoid((threshold-margin)/T). "
                                       "<=0 keeps hard gating behavior."))
        self.parser.add_argument("--late_interaction_rerank_soft_min_gate", type=float, default=0.0,
                                 help=("When soft gating is enabled, skip queries with gate <= this value. "
                                       "0 keeps all queries."))
        self.parser.add_argument("--late_interaction_preserve_baseline_top1", action="store_true",
                                 help=("Keep baseline top-1 video score as a floor after residual rerank. "
                                       "Useful to protect VR recall from over-aggressive residual updates."))
        self.parser.add_argument("--late_interaction_preserve_top1_margin", type=float, default=0.0,
                                 help=("When preserving baseline top-1, enforce top1 >= max(other reranked)+margin. "
                                       "0 keeps tie-level protection only."))
        self.parser.add_argument("--late_interaction_train_start_epoch", type=int, default=0,
                                 help="Enable late-interaction rerank in training starting from this epoch.")
        self.parser.add_argument("--late_interaction_train_score_weight", type=float, default=-1.0,
                                 help=("Train-time residual fusion weight for late interaction. "
                                       "<0 means using --late_interaction_score_weight."))
        self.parser.add_argument("--late_interaction_eval_score_weight", type=float, default=-1.0,
                                 help=("Eval-time residual fusion weight for late interaction. "
                                       "<0 means using --late_interaction_score_weight."))
        self.parser.add_argument("--late_interaction_residual_clip", type=float, default=-1.0,
                                 help=("Clip normalized late residual score to [-clip, clip]. "
                                       "<=0 disables clipping."))
        self.parser.add_argument("--late_interaction_rank_head_weight", type=float, default=1.0,
                                 help=("Position-aware residual weight at rank-1 within rerank top-k. "
                                       "1.0 disables rank protection; smaller values protect top ranks."))
        self.parser.add_argument("--late_interaction_rank_gamma", type=float, default=1.0,
                                 help=("Shape factor for position-aware residual weighting. "
                                       ">1 strengthens protection near top ranks."))
        self.parser.add_argument("--late_interaction_detach_backbone_in_train", action="store_true",
                                 help=("Detach encoded query/context from backbone for late branch in training "
                                       "(stabilizes baseline retrieval)."))
        self.parser.add_argument("--late_interaction_score_weight", type=float, default=0.2,
                                 help="Residual fusion weight for late interaction score.")
        self.parser.add_argument("--late_interaction_train_score_warmup_epochs", type=int, default=0,
                                 help=("Linearly warm up train-time late-interaction score weight. "
                                       "0 disables warmup."))
        self.parser.add_argument("--late_interaction_score_normalize", type=str, default="zscore",
                                 choices=["none", "zscore", "minmax"],
                                 help="Normalize late interaction residual scores before fusing with baseline.")
        self.parser.add_argument("--late_interaction_apply_to_vcl", action="store_true",
                                 help="Apply residual late rerank to VCL auxiliary loss branch (slower).")
        self.parser.add_argument("--multi_vector_query_max_count", type=int, default=6,
                                 help="Max number of content query vectors used in late interaction.")
        self.parser.add_argument("--multi_vector_phrase_window", type=int, default=1,
                                 help="Half-window size for local phrase pooling around selected query vectors.")
        self.parser.add_argument("--multi_vector_disable_phrase_pooling", action="store_true",
                                 help="Disable local phrase pooling and use selected content tokens directly.")
        self.parser.add_argument("--multi_vector_no_global_fallback", action="store_true",
                                 help="Disable adding one global fallback query vector.")

        self.parser.add_argument("--use_generative_augmentation", action="store_true",
                                 help="Enable decoder LM loss during training")
        self.parser.add_argument("--use_fusion_encoder", action="store_true",
                                 help="Enable query-video fusion encoder")
        self.parser.add_argument("--fusion_num_layers", type=int, default=2,
                                 help="number of fusion encoder layers")
        self.parser.add_argument("--lm_weight", type=float, default=0.3, help="weight for LM loss")
        self.parser.add_argument("--tokenizer_name_or_path", type=str, default="roberta-base")
        self.parser.add_argument("--lm_max_len", type=int, default=30, help="max length for LM inputs")
        self.parser.add_argument("--lm_start_token", type=str, default="[DEC]",
                                 help="start token for the decoder input sequence")
        self.parser.add_argument("--lm_start_token_id", type=int, default=None,
                                 help="override decoder start token id")
        self.parser.add_argument("--lm_disable_start_token", action="store_true",
                                 help="disable adding a decoder start token")
        self.parser.add_argument("--lm_vocab_size", type=int, default=None, help="override tokenizer vocab size")
        self.parser.add_argument("--lm_pad_token_id", type=int, default=None, help="override tokenizer pad id")
        self.parser.add_argument("--lm_num_layers", type=int, default=2, help="decoder layers for LM loss")

        self.parser.add_argument("--mamba_d_state", type=int, default=16)
        self.parser.add_argument("--mamba_d_conv", type=int, default=4)
        self.parser.add_argument("--mamba_expand", type=int, default=2)
        self.parser.add_argument("--mamba_fuse_mode", type=str, default="sum", choices=["sum", "concat"])

        # Semantic perturbation (default off, strict config-driven)
        self.parser.add_argument("--semantic_enable", action="store_true",
                                 help="Enable semantic perturbation training.")
        self.parser.add_argument("--semantic_backend", type=str, default="none", choices=["none", "llm"],
                                 help="Semantic backend: none|llm. No implicit fallback is allowed.")
        self.parser.add_argument("--semantic_strict_mode", action="store_true", default=True,
                                 help="Enable strict cache verification and fail-fast checks.")
        self.parser.add_argument("--semantic_no_strict_mode", action="store_true",
                                 help="Disable strict mode. Not recommended for reproducible experiments.")
        self.parser.add_argument("--semantic_no_fallback", action="store_true", default=True,
                                 help="Disallow any fallback path for Semantic.")
        self.parser.add_argument("--semantic_allow_fallback", action="store_true",
                                 help="Allow fallback behavior. Not recommended for ablations.")
        self.parser.add_argument("--semantic_cache_path", type=str, default="",
                                 help="Path to frozen Semantic cache jsonl file.")
        self.parser.add_argument("--semantic_fail_on_missing_cache", action="store_true", default=True,
                                 help="Fail when Semantic cache file is missing.")
        self.parser.add_argument("--semantic_allow_missing_cache", action="store_true",
                                 help="Do not fail on missing cache.")
        self.parser.add_argument("--semantic_fail_on_invalid_cache", action="store_true", default=True,
                                 help="Fail when Semantic cache content/metadata is invalid.")
        self.parser.add_argument("--semantic_allow_invalid_cache", action="store_true",
                                 help="Do not fail on invalid cache.")

        self.parser.add_argument("--semantic_build_cache_only", action="store_true",
                                 help="Only build Semantic cache and exit.")
        self.parser.add_argument("--semantic_cache_split", type=str, default="train",
                                 help="Split name embedded in Semantic cache metadata.")
        self.parser.add_argument("--semantic_num_hard_neg", type=int, default=2)
        self.parser.add_argument("--semantic_num_hard_pos", type=int, default=2)
        self.parser.add_argument("--semantic_max_retries_same_backend", type=int, default=2)
        self.parser.add_argument("--semantic_prompt_version", type=str, default="semantic_generator_v1")
        self.parser.add_argument("--semantic_schema_version", type=str, default="semantic_schema_v1")
        self.parser.add_argument("--semantic_generator_model", type=str, default="gpt-4.1-mini")
        self.parser.add_argument("--semantic_verifier_model", type=str, default="gpt-4.1-mini")
        self.parser.add_argument("--semantic_temperature", type=float, default=0.1)
        self.parser.add_argument("--semantic_seed", type=int, default=2018)
        self.parser.add_argument("--semantic_llm_api_base", type=str, default="")
        self.parser.add_argument("--semantic_llm_api_key", type=str, default="")
        self.parser.add_argument("--semantic_llm_transport", type=str, default="remote_api",
                                 choices=["remote_api", "local_xgrammar"])
        self.parser.add_argument("--semantic_llm_response_mode", type=str, default="json_schema",
                                 choices=["json_schema", "none"])
        self.parser.add_argument("--semantic_local_model_name_or_path", type=str, default="")
        self.parser.add_argument("--semantic_local_device", type=str, default="auto")
        self.parser.add_argument("--semantic_local_mask_backend", type=str, default="auto")
        self.parser.add_argument("--semantic_local_max_new_tokens", type=int, default=256)

        self.parser.add_argument("--semantic_neg_types", type=str, nargs="+",
                                 default=["attribute_swap", "action_swap", "role_swap",
                                          "temporal_order_flip", "count_state_swap", "object_scene_swap"])
        self.parser.add_argument("--semantic_pos_types", type=str, nargs="+",
                                 default=["paraphrase", "syntax_reorder", "modifier_compress", "lexical_variation"])
        self.parser.add_argument("--semantic_severity_levels", type=int, nargs="+", default=[1, 2, 3])

        self.parser.add_argument("--semantic_use_preference_loss", action="store_true",
                                 help="Enable Semantic preference loss.")
        self.parser.add_argument("--semantic_preference_margin", type=float, default=0.2)
        self.parser.add_argument("--semantic_preference_weight", type=float, default=1.0)
        self.parser.add_argument("--semantic_use_consistency_loss", action="store_true",
                                 help="Enable Semantic consistency loss.")
        self.parser.add_argument("--semantic_consistency_weight", type=float, default=1.0)
        self.parser.add_argument("--semantic_text_encoder_name_or_path", type=str, default="roberta-base",
                                 help="Text encoder for perturbation query features (Semantic only).")

        # Compositional supervision (rewrite-aware training on top of the main objective).
        self.parser.add_argument("--enable_compositional_supervision", action="store_true",
                                 help="Enable rewrite-aware compositional supervision on top of the base training loss.")
        self.parser.add_argument("--positive_rewrite_sample_size", type=int, default=2,
                                 help="Number of positive rewrites sampled per anchor query.")
        self.parser.add_argument("--negative_rewrite_sample_size", type=int, default=2,
                                 help="Number of negative rewrites sampled per anchor query.")
        self.parser.add_argument("--positive_invariance_weight", type=float, default=1.0,
                                 help="Weight for positive invariance supervision.")
        self.parser.add_argument("--negative_preference_weight", type=float, default=1.0,
                                 help="Weight for negative preference supervision.")
        self.parser.add_argument("--enable_debiased_retrieval_correction", action="store_true",
                                 help="Enable optional debiased correction for near-positive negatives.")
        self.parser.add_argument("--debiased_retrieval_weight", type=float, default=0.0,
                                 help="Weight for optional debiased correction loss.")
        self.parser.add_argument("--compositional_warmup_epochs", type=int, default=0,
                                 help="Number of warmup epochs before compositional supervision starts.")
        self.parser.add_argument("--compositional_ramp_epochs", type=int, default=3,
                                 help="Ramp-up epochs for compositional supervision weights.")
        self.parser.add_argument("--negative_preference_delay_epochs", type=int, default=1,
                                 help="Delay (after compositional warmup) before negative preference starts.")
        self.parser.add_argument("--debiased_retrieval_delay_epochs", type=int, default=2,
                                 help="Delay (after compositional warmup) before debiased correction starts.")
        self.parser.add_argument("--rewrite_type_quota_enabled", action="store_true", default=True,
                                 help="Enable diversity-aware rewrite type quota during sampling.")
        self.parser.add_argument("--rewrite_type_quota_disabled", action="store_true",
                                 help="Disable rewrite type quota.")
        self.parser.add_argument("--risky_negative_filter_enabled", action="store_true", default=True,
                                 help="Filter risky high-overlap temporal/role negatives.")
        self.parser.add_argument("--risky_negative_filter_disabled", action="store_true",
                                 help="Disable risky negative filtering.")
        self.parser.add_argument("--risky_negative_overlap_threshold", type=float, default=0.90,
                                 help="Overlap threshold for identifying risky negatives.")
        self.parser.add_argument("--risky_negative_start_epoch", type=int, default=0,
                                 help="Delay risky negatives until this epoch.")
        self.parser.add_argument("--risky_negative_downweight", type=float, default=0.5,
                                 help="Downweight factor for risky negatives when kept.")
        self.parser.add_argument("--collision_sanitization_enabled", action="store_true", default=True,
                                 help="Enable deterministic collision sanitization across positive/negative rewrites.")
        self.parser.add_argument("--collision_sanitization_disabled", action="store_true",
                                 help="Disable collision sanitization.")
        self.parser.add_argument("--allow_missing_rewrites", action="store_true",
                                 help="Allow training fallback when rewrite cache or rewrite entries are missing.")
        self.parser.add_argument("--require_rewrite_cache", action="store_true",
                                 help="Require rewrite cache presence and strict validation.")

        self.parser.add_argument("--min_pred_l", type=int, default=2,
                                 help="constrain the [st, ed] with ed - st >= 2 (2 clips with length 1.5 each, 3 secs "
                                      "in total this is the min length for proposal-based backup_method)")
        self.parser.add_argument("--max_pred_l", type=int, default=16,
                                 help="constrain the [st, ed] pairs with ed - st <= 16, 24 secs in total (16 clips "
                                      "with length 1.5 each, this is the max length for proposal-based backup_method)")
        self.parser.add_argument("--q2c_alpha", type=float, default=30,
                                 help="give more importance to top scored videos' spans,  "
                                      "the new score will be: s_new = exp(alpha * s), "
                                      "higher alpha indicates more importance. Note s in [-1, 1]")
        self.parser.add_argument("--q2c_alpha_vcmr", type=float, default=-1.0,
                                 help=("Override q2c alpha only for VCMR span scoring. "
                                       "<0 means using --q2c_alpha."))
        self.parser.add_argument("--vcmr_video_score_weight", type=float, default=1.0,
                                 help=("Weight for video score term in VCMR span scoring. "
                                       "1.0 keeps original behavior; lower values reduce video-score dominance."))
        self.parser.add_argument("--max_before_nms", type=int, default=200)
        self.parser.add_argument("--max_vcmr_video", type=int, default=100, help="re-ranking in top-max_vcmr_video")
        self.parser.add_argument("--export_score_diagnostics", action="store_true",
                                 help="Export per-query score diagnostics (baseline/fused/late/span terms) in eval.")
        self.parser.add_argument("--score_diagnostics_topk", type=int, default=10,
                                 help="How many top-ranked videos to keep per query in score diagnostics export.")
        self.parser.add_argument("--score_diagnostics_filename", type=str, default="",
                                 help=("Optional diagnostics filename. Empty means auto name based on submission file. "
                                       "Saved under results_dir."))
        self.parser.add_argument("--nms_thd", type=float, default=-1,
                                 help="additionally use non-maximum suppression (or non-minimum suppression for "
                                      "distance) to post-processing the predictions. -1: do not use nms. 0.6 for "
                                      "charades_sta, 0.5 for anet_cap")

    def display_save(self, opt):
        args = vars(opt)

        print("------------ Options -------------\n{}\n-------------------".format({str(k): str(v) for k, v in
                                                                                    sorted(args.items())}))

        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)
            save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.no_core_driver = True
            opt.num_workers = 0
            opt.eval_query_bsz = 100
        if isinstance(self, TestOptions):

            opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",
                               "eval_split_name", "eval_path", "eval_query_bsz", "eval_context_bsz",
                               "max_pred_l", "min_pred_l", "external_inference_vr_res_path",
                               "export_score_diagnostics", "score_diagnostics_topk", "score_diagnostics_filename"]:
                    setattr(opt, arg, saved_options[arg])
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")
            if opt.clip_length is None:
                opt.clip_length = ProposalConfigs[opt.dset_name]["clip_length"]
                print("Loaded clip_length {} from proposal config file".format(opt.clip_length))
            opt.results_dir = os.path.join(opt.results_root, "-".join([opt.dset_name, opt.ctx_mode, opt.exp_id,
                                                                       time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)

            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename, enclosing_dir="code", exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"],)
        self.display_save(opt)
        if getattr(opt, "use_generative_augmentation", False) and not getattr(opt, "use_fusion_encoder", False):
            raise ValueError("--use_fusion_encoder is required when --use_generative_augmentation is enabled.")
        if getattr(opt, "use_generative_augmentation", False):
            if opt.lm_max_len < 2:
                raise ValueError("--lm_max_len must be >= 2 for LM loss.")
            if opt.lm_max_len > opt.max_desc_l:
                raise ValueError("--lm_max_len must be <= --max_desc_l to match decoder positional embeddings.")
        if opt.hard_negative_start_epoch != -1:
            if opt.hard_pool_size > opt.bsz:
                print("[WARNING] hard_pool_size is larger than bsz")

        # Normalize Semantic strict toggles.
        if getattr(opt, "semantic_no_strict_mode", False):
            opt.semantic_strict_mode = False
        if getattr(opt, "semantic_allow_fallback", False):
            opt.semantic_no_fallback = False
        if getattr(opt, "semantic_allow_missing_cache", False):
            opt.semantic_fail_on_missing_cache = False
        if getattr(opt, "semantic_allow_invalid_cache", False):
            opt.semantic_fail_on_invalid_cache = False

        if getattr(opt, "rewrite_type_quota_disabled", False):
            opt.rewrite_type_quota_enabled = False
        if getattr(opt, "risky_negative_filter_disabled", False):
            opt.risky_negative_filter_enabled = False
        if getattr(opt, "collision_sanitization_disabled", False):
            opt.collision_sanitization_enabled = False

        if getattr(opt, "enable_compositional_supervision", False):
            opt.semantic_enable = True
            if opt.semantic_backend == "none":
                opt.semantic_backend = "llm"
            opt.semantic_num_hard_pos = int(opt.positive_rewrite_sample_size)
            opt.semantic_num_hard_neg = int(opt.negative_rewrite_sample_size)
            opt.semantic_use_consistency_loss = opt.positive_invariance_weight > 0
            opt.semantic_use_preference_loss = opt.negative_preference_weight > 0
            opt.semantic_consistency_weight = float(opt.positive_invariance_weight)
            opt.semantic_preference_weight = float(opt.negative_preference_weight)
            if not bool(opt.require_rewrite_cache):
                opt.allow_missing_rewrites = True
                opt.semantic_no_fallback = False
                opt.semantic_fail_on_missing_cache = False
                opt.semantic_fail_on_invalid_cache = False

        if opt.semantic_num_hard_neg < 0 or opt.semantic_num_hard_pos < 0:
            raise ValueError("--semantic_num_hard_neg and --semantic_num_hard_pos must be >= 0.")
        if opt.positive_rewrite_sample_size < 0 or opt.negative_rewrite_sample_size < 0:
            raise ValueError("--positive_rewrite_sample_size and --negative_rewrite_sample_size must be >= 0.")
        if opt.semantic_max_retries_same_backend < 0:
            raise ValueError("--semantic_max_retries_same_backend must be >= 0.")
        if opt.semantic_temperature < 0:
            raise ValueError("--semantic_temperature must be >= 0.")
        if opt.semantic_local_max_new_tokens <= 0:
            raise ValueError("--semantic_local_max_new_tokens must be > 0.")
        if opt.compositional_warmup_epochs < 0 or opt.compositional_ramp_epochs < 0:
            raise ValueError("--compositional_warmup_epochs and --compositional_ramp_epochs must be >= 0.")
        if opt.negative_preference_delay_epochs < 0 or opt.debiased_retrieval_delay_epochs < 0:
            raise ValueError("--negative_preference_delay_epochs and --debiased_retrieval_delay_epochs must be >= 0.")
        if not (0.0 <= opt.risky_negative_overlap_threshold <= 1.0):
            raise ValueError("--risky_negative_overlap_threshold must be in [0, 1].")
        if opt.risky_negative_downweight < 0:
            raise ValueError("--risky_negative_downweight must be >= 0.")
        if opt.risky_negative_start_epoch < 0:
            raise ValueError("--risky_negative_start_epoch must be >= 0.")
        if any(e not in [1, 2, 3] for e in opt.semantic_severity_levels):
            raise ValueError("--semantic_severity_levels only support 1/2/3.")

        if opt.semantic_enable:
            if opt.semantic_backend == "none":
                raise ValueError("semantic_enable=true requires --semantic_backend=llm.")
            if opt.semantic_backend != "llm":
                raise ValueError("Unsupported --semantic_backend={}. Only llm is supported.".format(opt.semantic_backend))
            if len(opt.device_ids) > 1:
                raise ValueError("Semantic currently supports single-GPU training only. Set --device_ids to one GPU.")
            if (not opt.semantic_build_cache_only) and (
                not opt.semantic_use_preference_loss
                and not opt.semantic_use_consistency_loss
                and not opt.enable_debiased_retrieval_correction
            ):
                raise ValueError(
                    "semantic_enable=true requires at least one of "
                    "--semantic_use_preference_loss / --semantic_use_consistency_loss / --enable_debiased_retrieval_correction."
                )
            if opt.semantic_num_hard_neg > 0 and len(opt.semantic_neg_types) == 0:
                raise ValueError("--semantic_neg_types must be non-empty when --semantic_num_hard_neg > 0.")
            if opt.semantic_num_hard_pos > 0 and len(opt.semantic_pos_types) == 0:
                raise ValueError("--semantic_pos_types must be non-empty when --semantic_num_hard_pos > 0.")
            if opt.semantic_no_fallback:
                # In strict no-fallback mode, missing/invalid cache must always fail.
                opt.semantic_fail_on_missing_cache = True
                opt.semantic_fail_on_invalid_cache = True
            if not opt.semantic_build_cache_only and not opt.semantic_cache_path and not bool(opt.allow_missing_rewrites):
                raise ValueError("semantic_enable=true requires --semantic_cache_path unless --allow_missing_rewrites is set.")
            if opt.semantic_build_cache_only and opt.semantic_llm_transport == "local_xgrammar":
                if not str(opt.semantic_local_model_name_or_path).strip():
                    raise ValueError(
                        "semantic_llm_transport=local_xgrammar requires --semantic_local_model_name_or_path."
                    )
        else:
            if opt.semantic_build_cache_only:
                raise ValueError("--semantic_build_cache_only requires --semantic_enable.")

        if opt.retrieval_scorer in {"late_interaction", "combined"}:
            opt.retrieval_scorer = "residual_rerank"

        opt.late_interaction_use_projection = not bool(getattr(opt, "late_interaction_no_projection", False))
        if opt.late_interaction_dim < 0:
            raise ValueError("--late_interaction_dim must be >= 0.")
        if opt.late_interaction_video_chunk_size <= 0:
            raise ValueError("--late_interaction_video_chunk_size must be > 0.")
        if opt.late_interaction_token_weight_floor < 0:
            raise ValueError("--late_interaction_token_weight_floor must be >= 0.")
        if opt.late_interaction_rerank_topk < 0:
            raise ValueError("--late_interaction_rerank_topk must be >= 0.")
        if opt.late_interaction_train_rerank_topk < 0:
            opt.late_interaction_train_rerank_topk = opt.late_interaction_rerank_topk
        if opt.late_interaction_eval_rerank_topk < 0:
            opt.late_interaction_eval_rerank_topk = opt.late_interaction_rerank_topk
        if opt.late_interaction_train_rerank_topk < 0:
            raise ValueError("--late_interaction_train_rerank_topk must be >= 0.")
        if opt.late_interaction_eval_rerank_topk < 0:
            raise ValueError("--late_interaction_eval_rerank_topk must be >= 0.")
        if opt.late_interaction_query_chunk_size <= 0:
            raise ValueError("--late_interaction_query_chunk_size must be > 0.")
        if opt.late_interaction_rerank_margin_threshold < 0:
            opt.late_interaction_rerank_margin_threshold = -1.0
        if opt.late_interaction_rerank_soft_temperature < 0:
            raise ValueError("--late_interaction_rerank_soft_temperature must be >= 0.")
        if not (0.0 <= opt.late_interaction_rerank_soft_min_gate <= 1.0):
            raise ValueError("--late_interaction_rerank_soft_min_gate must be in [0, 1].")
        if opt.late_interaction_preserve_top1_margin < 0:
            raise ValueError("--late_interaction_preserve_top1_margin must be >= 0.")
        if opt.late_interaction_train_start_epoch < 0:
            raise ValueError("--late_interaction_train_start_epoch must be >= 0.")
        if opt.late_interaction_train_score_weight < 0:
            opt.late_interaction_train_score_weight = opt.late_interaction_score_weight
        if opt.late_interaction_eval_score_weight < 0:
            opt.late_interaction_eval_score_weight = opt.late_interaction_score_weight
        if not (0.0 <= opt.late_interaction_score_weight <= 1.0):
            raise ValueError("--late_interaction_score_weight must be in [0, 1].")
        if not (0.0 <= opt.late_interaction_train_score_weight <= 1.0):
            raise ValueError("--late_interaction_train_score_weight must be in [0, 1].")
        if not (0.0 <= opt.late_interaction_eval_score_weight <= 1.0):
            raise ValueError("--late_interaction_eval_score_weight must be in [0, 1].")
        if not (0.0 < opt.late_interaction_rank_head_weight <= 1.0):
            raise ValueError("--late_interaction_rank_head_weight must be in (0, 1].")
        if opt.late_interaction_rank_gamma <= 0:
            raise ValueError("--late_interaction_rank_gamma must be > 0.")
        if opt.late_interaction_train_score_warmup_epochs < 0:
            raise ValueError("--late_interaction_train_score_warmup_epochs must be >= 0.")
        if opt.multi_vector_query_max_count <= 0:
            raise ValueError("--multi_vector_query_max_count must be > 0.")
        if opt.multi_vector_phrase_window < 0:
            raise ValueError("--multi_vector_phrase_window must be >= 0.")
        opt.multi_vector_use_phrase_pooling = not bool(getattr(opt, "multi_vector_disable_phrase_pooling", False))
        opt.multi_vector_use_global_fallback = not bool(getattr(opt, "multi_vector_no_global_fallback", False))
        if opt.q2c_alpha_vcmr < 0:
            opt.q2c_alpha_vcmr = opt.q2c_alpha
        if opt.vcmr_video_score_weight <= 0:
            raise ValueError("--vcmr_video_score_weight must be > 0.")
        if opt.score_diagnostics_topk <= 0:
            raise ValueError("--score_diagnostics_topk must be > 0.")
        if opt.early_stop_min_delta < 0:
            raise ValueError("--early_stop_min_delta must be >= 0.")
        if opt.early_stop_vcmr_weight < 0 or opt.early_stop_svmr_weight < 0 or opt.early_stop_vr_weight < 0:
            raise ValueError("Composite early-stop weights must be >= 0.")
        if opt.early_stop_vcmr_scale <= 0 or opt.early_stop_svmr_scale <= 0 or opt.early_stop_vr_scale <= 0:
            raise ValueError("Composite early-stop scales must be > 0.")
        if not opt.early_stop_vcmr_metrics:
            raise ValueError("--early_stop_vcmr_metrics must not be empty.")
        if not opt.early_stop_svmr_metrics:
            raise ValueError("--early_stop_svmr_metrics must not be empty.")
        if not opt.early_stop_vr_metrics:
            raise ValueError("--early_stop_vr_metrics must not be empty.")

        assert opt.stop_task in opt.eval_tasks_at_training
        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        opt.h5driver = None if opt.no_core_driver else "core"

        opt.num_workers = 1 if opt.no_core_driver else opt.num_workers
        opt.pin_memory = not opt.no_pin_memory
        if "video" in opt.ctx_mode and opt.vid_feat_size > 3000:
            assert opt.no_norm_vfeat
        if "tef" in opt.ctx_mode and "video" in opt.ctx_mode:
            opt.vid_feat_size += 2
        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--tasks", type=str, nargs="+",
                                 choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"],
                                 help="Which tasks to run."
                                      "VCMR: Video Corpus Moment Retrieval;"
                                      "SVMR: Single Video Moment Retrieval;"
                                      "VR: regular Video Retrieval. (will be performed automatically with VCMR)")
