from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked_scores = scores.masked_fill(mask == 0, -1e10)
    return F.softmax(masked_scores, dim=dim)


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    size = int(size) // 2
    x = np.arange(-size, size + 1)
    normal = 1 / (np.sqrt(2.0 * np.pi) * sigma)
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2)) * normal
    gkernel = torch.from_numpy(g).float().to(device).view(1, 1, -1)
    return gkernel


def get_dynamic_scores(scores: torch.Tensor, stride: int, masks: torch.Tensor, ths: float = 0.0005,
                       sigma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    def nchk(f, f1, f2, ths_):
        return (((3 * f) > ths_) | ((2 * f + f1) > ths_) | ((f + f1 + f2) > ths_))

    gstride = min(stride - 2, 3)
    if stride < 3:
        gkernel = torch.ones((1, 1, 1), device=scores.device)
    else:
        gkernel = _gaussian_kernel(gstride, sigma, scores.device)
    gscore = F.conv1d(scores.view(-1, 1, scores.size(-1)), gkernel).view(scores.size(0), -1)

    diffres = torch.diff(gscore)
    pad_left = torch.zeros((diffres.size(0), (masks.size(-1) - diffres.size(-1)) // 2), device=scores.device)
    pad_right = torch.zeros((diffres.size(0),
                             masks.size(-1) - diffres.size(-1) - pad_left.size(-1)), device=scores.device)
    diffres = torch.cat((pad_left, diffres, pad_right), dim=-1) * masks

    dynamic_scores = torch.zeros_like(diffres)
    dynamic_idxs = torch.zeros_like(diffres)
    for idx in range(diffres.size(0)):
        f1 = f2 = f3 = 0.0
        d_score = 0.0
        d_idx = 0
        for i in range(diffres.size(-1)):
            f3 = f2
            f2 = f1
            f1 = diffres[idx][i].item()
            if nchk(f1, f2, f3, ths):
                d_score += max(3 * f1, 2 * f1 + f2, f1 + f2 + f3)
            else:
                d_idx = i
                d_score = 0.0
            dynamic_idxs[idx][i] = d_idx / scores.size(-1)
            dynamic_scores[idx][i] = d_score
    return dynamic_idxs, dynamic_scores


def compute_tfvtg_st_ed_probs(temporal_curve: torch.Tensor, video_mask: torch.Tensor, stride: int, max_stride: int,
                              dynamic_weight: float = 1.0, static_weight: float = 1.0,
                              smooth_win: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        temporal_curve: (B, L)
        video_mask: (B, L)
    """
    scores = temporal_curve * video_mask
    stride = max(1, min(stride, scores.size(-1) // 2))
    max_stride = max(stride, min(max_stride, scores.size(-1)))
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, video_mask, sigma=smooth_win)

    bsz, seq_l = scores.size(0), scores.size(1)
    start_scores = scores.new_zeros(bsz, seq_l)
    end_scores = scores.new_zeros(bsz, seq_l)

    for kernel_size in range(stride, min(seq_l + 1, max_stride + 1), stride):
        kernel = torch.ones((1, 1, kernel_size), device=scores.device)
        inner_sum = F.conv1d(scores.view(-1, 1, seq_l), kernel).view(bsz, -1)
        inner_num = F.conv1d(video_mask.view(-1, 1, seq_l), kernel).view(bsz, -1)
        outer_sum = (scores * video_mask).sum(dim=-1, keepdim=True) - inner_sum
        outer_num = video_mask.sum(dim=-1, keepdim=True) - inner_num
        outer_num = outer_num.clamp(min=1.0)
        static_scores = inner_sum / kernel_size - outer_sum / outer_num
        static_scores = static_scores.masked_fill(inner_num == 0, -1e3)
        valid_mask = static_scores > -1e3
        static_scores = static_scores.masked_fill(~valid_mask, 0.0)

        positions = torch.arange(0, static_scores.size(-1), device=scores.device)
        end_idx = positions + kernel_size - 1
        end_idx = end_idx.clamp(max=seq_l - 1).unsqueeze(0).expand(bsz, -1)
        dyn_idx = dynamic_idxs.narrow(-1, 0, static_scores.size(-1))
        dyn_idx = (dyn_idx * seq_l).round().long().clamp(min=0, max=seq_l - 1)
        dyn_scores = dynamic_scores.narrow(-1, 0, static_scores.size(-1))
        dyn_scores = dyn_scores.masked_fill(~valid_mask, 0.0)

        proposal_scores = static_scores * static_weight + dyn_scores * dynamic_weight

        end_scores.scatter_add_(1, end_idx, proposal_scores)
        start_scores.scatter_add_(1, dyn_idx, proposal_scores)

    if not torch.isfinite(start_scores).any():
        start_scores = scores
    if not torch.isfinite(end_scores).any():
        end_scores = scores
    st_probs = _masked_softmax(start_scores, video_mask)
    ed_probs = _masked_softmax(end_scores, video_mask)
    return st_probs, ed_probs
