from typing import List

import torch
import torch.nn.functional as F


def severity_to_weight(severity: int) -> float:
    level = int(severity)
    if level < 1:
        level = 1
    if level > 3:
        level = 3
    return 1.0 + 0.5 * float(level - 1)


def compute_preference_loss(
    anchor_scores: torch.Tensor,
    pos_scores_per_sample: List[torch.Tensor],
    neg_scores_per_sample: List[torch.Tensor],
    neg_weights_per_sample: List[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    total = anchor_scores.new_tensor(0.0)
    count = 0

    for idx, anchor_score in enumerate(anchor_scores):
        neg_scores = neg_scores_per_sample[idx]
        neg_weights = neg_weights_per_sample[idx]
        pos_scores = pos_scores_per_sample[idx]

        if neg_scores.numel() > 0:
            loss_neg = F.relu(margin + neg_scores - anchor_score)
            if neg_weights.numel() == neg_scores.numel():
                total = total + torch.sum(loss_neg * neg_weights)
                count += int(torch.sum(neg_weights > 0).item())
            else:
                total = total + torch.sum(loss_neg)
                count += int(neg_scores.numel())

        if neg_scores.numel() > 0 and pos_scores.numel() > 0:
            half_margin = margin * 0.5
            for pos in pos_scores:
                loss_pos_vs_neg = F.relu(half_margin + neg_scores - pos)
                if neg_weights.numel() == neg_scores.numel():
                    total = total + torch.sum(loss_pos_vs_neg * neg_weights)
                    count += int(torch.sum(neg_weights > 0).item())
                else:
                    total = total + torch.sum(loss_pos_vs_neg)
                    count += int(neg_scores.numel())

    if count == 0:
        return anchor_scores.new_tensor(0.0)
    return total / float(count)


def compute_consistency_loss(anchor_scores: torch.Tensor, pos_scores_per_sample: List[torch.Tensor]) -> torch.Tensor:
    total = anchor_scores.new_tensor(0.0)
    count = 0
    for idx, anchor_score in enumerate(anchor_scores):
        pos_scores = pos_scores_per_sample[idx]
        if pos_scores.numel() == 0:
            continue
        total = total + torch.sum((pos_scores - anchor_score).pow(2))
        count += int(pos_scores.numel())
    if count == 0:
        return anchor_scores.new_tensor(0.0)
    return total / float(count)


def compute_debiased_correction_loss(
    anchor_scores: torch.Tensor,
    neg_scores_per_sample: List[torch.Tensor],
    debias_weights_per_sample: List[torch.Tensor],
) -> torch.Tensor:
    """Pull near-positive negatives closer to anchor scores with a lightweight correction term."""
    total = anchor_scores.new_tensor(0.0)
    count = 0
    for idx, anchor_score in enumerate(anchor_scores):
        neg_scores = neg_scores_per_sample[idx]
        debias_weights = debias_weights_per_sample[idx]
        if neg_scores.numel() == 0 or debias_weights.numel() == 0:
            continue
        if neg_scores.numel() != debias_weights.numel():
            continue
        total = total + torch.sum((anchor_score - neg_scores).pow(2) * debias_weights)
        count += int(torch.sum(debias_weights > 0).item())
    if count == 0:
        return anchor_scores.new_tensor(0.0)
    return total / float(count)
