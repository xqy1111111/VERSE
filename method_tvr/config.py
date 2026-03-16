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

        self.parser.add_argument("--use_generative_augmentation", action="store_true",
                                 help="Enable decoder LM loss during training")
        self.parser.add_argument("--use_fusion_encoder", action="store_true",
                                 help="Enable query-video fusion encoder")
        self.parser.add_argument("--fusion_num_layers", type=int, default=2,
                                 help="number of fusion encoder layers")
        self.parser.add_argument("--lm_weight", type=float, default=0.3, help="weight for LM loss")
        self.parser.add_argument("--tokenizer_name_or_path", type=str, default="bert-base-uncased")
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
        self.parser.add_argument("--semantic_text_encoder_name_or_path", type=str, default="bert-base-uncased",
                                 help="Text encoder for perturbation query features (Semantic only).")

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
        self.parser.add_argument("--max_before_nms", type=int, default=200)
        self.parser.add_argument("--max_vcmr_video", type=int, default=100, help="re-ranking in top-max_vcmr_video")
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
                               "max_pred_l", "min_pred_l", "external_inference_vr_res_path"]:
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

        if opt.semantic_num_hard_neg < 0 or opt.semantic_num_hard_pos < 0:
            raise ValueError("--semantic_num_hard_neg and --semantic_num_hard_pos must be >= 0.")
        if opt.semantic_max_retries_same_backend < 0:
            raise ValueError("--semantic_max_retries_same_backend must be >= 0.")
        if opt.semantic_temperature < 0:
            raise ValueError("--semantic_temperature must be >= 0.")
        if opt.semantic_local_max_new_tokens <= 0:
            raise ValueError("--semantic_local_max_new_tokens must be > 0.")
        if any(e not in [1, 2, 3] for e in opt.semantic_severity_levels):
            raise ValueError("--semantic_severity_levels only support 1/2/3.")

        if opt.semantic_enable:
            if opt.semantic_backend == "none":
                raise ValueError("semantic_enable=true requires --semantic_backend=llm.")
            if opt.semantic_backend != "llm":
                raise ValueError("Unsupported --semantic_backend={}. Only llm is supported.".format(opt.semantic_backend))
            if len(opt.device_ids) > 1:
                raise ValueError("Semantic currently supports single-GPU training only. Set --device_ids to one GPU.")
            if (not opt.semantic_build_cache_only) and (not opt.semantic_use_preference_loss and not opt.semantic_use_consistency_loss):
                raise ValueError("semantic_enable=true requires at least one of "
                                 "--semantic_use_preference_loss / --semantic_use_consistency_loss.")
            if opt.semantic_num_hard_neg > 0 and len(opt.semantic_neg_types) == 0:
                raise ValueError("--semantic_neg_types must be non-empty when --semantic_num_hard_neg > 0.")
            if opt.semantic_num_hard_pos > 0 and len(opt.semantic_pos_types) == 0:
                raise ValueError("--semantic_pos_types must be non-empty when --semantic_num_hard_pos > 0.")
            if opt.semantic_no_fallback:
                # In strict no-fallback mode, missing/invalid cache must always fail.
                opt.semantic_fail_on_missing_cache = True
                opt.semantic_fail_on_invalid_cache = True
            if not opt.semantic_build_cache_only and not opt.semantic_cache_path:
                raise ValueError("semantic_enable=true requires --semantic_cache_path for training.")
            if opt.semantic_build_cache_only and opt.semantic_llm_transport == "local_xgrammar":
                if not str(opt.semantic_local_model_name_or_path).strip():
                    raise ValueError(
                        "semantic_llm_transport=local_xgrammar requires --semantic_local_model_name_or_path."
                    )
        else:
            if opt.semantic_build_cache_only:
                raise ValueError("--semantic_build_cache_only requires --semantic_enable.")

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
