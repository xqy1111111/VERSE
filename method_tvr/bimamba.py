from typing import Optional

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    Mamba = None
    _mamba_import_error = exc


def _normalize_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 3:
        return attention_mask.squeeze(1)
    return attention_mask


def _reverse_by_length(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, dim = x.shape
    idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)  # (B, L)

    reversed_idx = torch.where(
        idx < lengths.unsqueeze(1),
        lengths.unsqueeze(1) - 1 - idx,
        idx
    )

    reversed_idx = reversed_idx.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(x, 1, reversed_idx)


class BiMambaEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, d_state=16, d_conv=4, expand=2, fuse_mode="sum"):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is required for BiMamba") from _mamba_import_error
        if fuse_mode not in ["sum", "concat"]:
            raise ValueError("fuse_mode must be 'sum' or 'concat'")
        self.fuse_mode = fuse_mode
        self.norm = nn.LayerNorm(hidden_size)
        self.mamba_f = Mamba(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_b = Mamba(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        if fuse_mode == "concat":
            self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_tensor: torch.Tensor,
                attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        mask = _normalize_mask(attention_mask)
        if mask is None:
            lengths = None
        else:
            lengths = mask.long().sum(dim=1)

        # Pre-norm
        x = self.norm(input_tensor)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        y_f = self.mamba_f(x)

        if lengths is None:
            y_b = self.mamba_b(torch.flip(x, dims=[1]))
            y_b = torch.flip(y_b, dims=[1])
        else:
            x_rev = _reverse_by_length(x, lengths)
            y_b = self.mamba_b(x_rev)
            y_b = _reverse_by_length(y_b, lengths)

        if self.fuse_mode == "sum":
            y = y_f + y_b
        else:
            y = self.proj(torch.cat([y_f, y_b], dim=-1))

        if mask is not None:
            y = y * mask.unsqueeze(-1)

        y = self.dropout(y)

        # Residual connection
        x = input_tensor + y

        # FFN with pre-norm
        y = self.ffn_norm(x)
        y = self.ffn(y)

        out = x + y
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out
