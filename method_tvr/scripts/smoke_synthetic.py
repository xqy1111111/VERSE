import argparse

import torch
from easydict import EasyDict as EDict

from method_tvr.model import ReLoCLNet
from method_tvr.tfvtg_scoring import compute_tfvtg_st_ed_probs


def build_config(args):
    cfg = EDict(
        visual_input_size=64,
        sub_input_size=2,
        query_input_size=64,
        hidden_size=32,
        conv_kernel_size=3,
        conv_stride=1,
        max_ctx_l=12,
        max_desc_l=8,
        input_drop=0.1,
        drop=0.1,
        n_heads=4,
        initializer_range=0.02,
        ctx_mode="video",
        margin=0.1,
        ranking_loss_type="hinge",
        lw_neg_q=1.0,
        lw_neg_ctx=1.0,
        lw_fcl=0.03,
        lw_vcl=0.03,
        lw_st_ed=0.01,
        use_hard_negative=False,
        hard_pool_size=10,
        use_sub=False,
        backbone_type=args.backbone_type,
        use_generative_augmentation=args.use_generative_augmentation,
        use_fusion_encoder=args.use_fusion_encoder or args.use_generative_augmentation,
        fusion_num_layers=2,
        lm_weight=0.1,
        lm_vocab_size=128 if args.use_generative_augmentation else None,
        lm_pad_token_id=0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_fuse_mode="sum",
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_type", type=str, default="Transformer",
                        choices=["Transformer", "BiMamba"])
    parser.add_argument("--use_generative_augmentation", action="store_true")
    parser.add_argument("--use_fusion_encoder", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this smoke test")
    device = torch.device("cuda")

    cfg = build_config(args)
    model = ReLoCLNet(cfg).to(device)
    model.train()

    bsz, lq, lv = 2, 8, 12
    query_feat = torch.randn(bsz, lq, cfg.query_input_size, device=device)
    query_mask = torch.ones(bsz, lq, device=device)
    video_feat = torch.randn(bsz, lv, cfg.visual_input_size, device=device)
    video_mask = torch.ones(bsz, lv, device=device)
    sub_feat = torch.zeros(bsz, 2, 2, device=device)
    sub_mask = torch.zeros(bsz, 2, device=device)
    st_ed_indices = torch.tensor([[1, 4], [2, 6]], device=device)
    match_labels = torch.zeros(bsz, lv, dtype=torch.long, device=device)
    match_labels[:, 1:6] = 1

    query_input_ids = None
    query_attn_mask = None
    if args.use_generative_augmentation:
        query_input_ids = torch.randint(0, cfg.lm_vocab_size, (bsz, lq), device=device)
        query_attn_mask = torch.ones(bsz, lq, dtype=torch.long, device=device)

    loss, loss_dict = model(query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask,
                            st_ed_indices, match_labels, query_input_ids, query_attn_mask)
    assert torch.isfinite(loss).all()

    model.eval()
    with torch.no_grad():
        _, _, _, _, x_video_feat, _ = model.encode_context(video_feat, video_mask, sub_feat, sub_mask,
                                                           return_mid_output=True)
        q2c_scores, st_prob, ed_prob, encoded_query = model.get_pred_from_raw_query(
            query_feat, query_mask, x_video_feat, video_mask, None, None, cross=False, return_encoded_query=True)
        assert st_prob.shape == (bsz, lv)
        assert ed_prob.shape == (bsz, lv)
        temporal_curve = model.get_temporal_curve(encoded_query, query_mask, x_video_feat, video_mask)
        st_tf, ed_tf = compute_tfvtg_st_ed_probs(temporal_curve, video_mask, stride=2, max_stride=6)
        assert st_tf.shape == (bsz, lv)
        assert ed_tf.shape == (bsz, lv)
    print("smoke_synthetic: OK")


if __name__ == "__main__":
    main()
