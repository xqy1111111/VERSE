import torch
import torch.nn.functional as F


def _masked_mean_pool(sequence_feat, sequence_mask):
    weights = sequence_mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    pooled = (sequence_feat * weights).sum(dim=1) / denom
    return F.normalize(pooled, dim=-1)


def _build_gap_weights(
    score_matrix,
    gap_threshold,
    gap_temperature,
    min_negative_weight,
):
    pos_scores = torch.diagonal(score_matrix, offset=0).unsqueeze(1)
    gaps = pos_scores - score_matrix
    temp = max(float(gap_temperature), 1e-6)
    gate = torch.sigmoid((gaps - float(gap_threshold)) / temp)
    return float(min_negative_weight) + (1.0 - float(min_negative_weight)) * gate


def _build_background_weights(
    video_feat,
    video_mask,
    background_similarity_threshold,
    background_temperature,
    background_downweight,
    min_negative_weight,
):
    pooled_video = _masked_mean_pool(video_feat, video_mask)
    bg_similarity = torch.matmul(pooled_video, pooled_video.t())
    temp = max(float(background_temperature), 1e-6)
    bg_gate = torch.sigmoid((bg_similarity - float(background_similarity_threshold)) / temp)
    bg_weights = 1.0 - float(background_downweight) * bg_gate
    return torch.clamp(bg_weights, min=float(min_negative_weight), max=1.0)


def _directional_weighted_nce(score_matrix, neg_weights, temperature):
    bsz = score_matrix.size(0)
    if bsz <= 1:
        return score_matrix.new_tensor(0.0)
    temp = max(float(temperature), 1e-6)
    logits = score_matrix / temp
    row_shift = torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(logits - row_shift)
    numerator = torch.diagonal(exp_logits, offset=0)
    denominator = numerator + torch.sum(exp_logits * neg_weights, dim=1)
    safe_ratio = numerator / denominator.clamp(min=1e-8)
    return (-torch.log(safe_ratio.clamp(min=1e-8))).mean()


def compute_debiased_video_frame_loss(
    query_context_scores,
    video_feat,
    video_mask,
    temperature=0.07,
    gap_threshold=0.05,
    gap_temperature=0.05,
    min_negative_weight=0.05,
    background_similarity_threshold=0.6,
    background_temperature=0.1,
    background_downweight=0.5,
):
    bsz = query_context_scores.size(0)
    if bsz <= 1:
        return query_context_scores.new_tensor(0.0)

    diag_mask = torch.eye(bsz, dtype=query_context_scores.dtype, device=query_context_scores.device)

    with torch.no_grad():
        gap_weights_q2v = _build_gap_weights(
            score_matrix=query_context_scores.detach(),
            gap_threshold=gap_threshold,
            gap_temperature=gap_temperature,
            min_negative_weight=min_negative_weight,
        )
        gap_weights_v2q = _build_gap_weights(
            score_matrix=query_context_scores.detach().t(),
            gap_threshold=gap_threshold,
            gap_temperature=gap_temperature,
            min_negative_weight=min_negative_weight,
        )

        bg_weights = _build_background_weights(
            video_feat=video_feat.detach(),
            video_mask=video_mask.detach(),
            background_similarity_threshold=background_similarity_threshold,
            background_temperature=background_temperature,
            background_downweight=background_downweight,
            min_negative_weight=min_negative_weight,
        )

        neg_weights_q2v = gap_weights_q2v * bg_weights * (1.0 - diag_mask)
        neg_weights_v2q = gap_weights_v2q * bg_weights.t() * (1.0 - diag_mask)

    loss_q2v = _directional_weighted_nce(
        score_matrix=query_context_scores,
        neg_weights=neg_weights_q2v,
        temperature=temperature,
    )
    loss_v2q = _directional_weighted_nce(
        score_matrix=query_context_scores.t(),
        neg_weights=neg_weights_v2q,
        temperature=temperature,
    )
    return 0.5 * (loss_q2v + loss_v2q)
