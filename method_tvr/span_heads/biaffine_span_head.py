import torch
import torch.nn as nn


def _mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class BiaffineSpanHead(nn.Module):
    """Biaffine joint span scorer over temporal tokens."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        self.start_proj = nn.Linear(1, hidden_size)
        self.end_proj = nn.Linear(1, hidden_size)
        self.start_bias = nn.Linear(hidden_size, 1, bias=False)
        self.end_bias = nn.Linear(hidden_size, 1, bias=False)
        self.biaffine_weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.biaffine_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.start_proj.weight)
        nn.init.zeros_(self.start_proj.bias)
        nn.init.xavier_uniform_(self.end_proj.weight)
        nn.init.zeros_(self.end_proj.bias)
        nn.init.xavier_uniform_(self.start_bias.weight)
        nn.init.xavier_uniform_(self.end_bias.weight)
        nn.init.xavier_uniform_(self.biaffine_weight)
        nn.init.zeros_(self.biaffine_bias)

    @staticmethod
    def _build_pair_mask(token_mask):
        pair_mask = torch.einsum("bl,bm->blm", token_mask, token_mask)
        return torch.triu(pair_mask, diagonal=0)

    def _forward_single(self, similarity, token_mask):
        similarity = similarity.unsqueeze(-1)
        start_repr = self.dropout(self.activation(self.start_proj(similarity)))
        end_repr = self.dropout(self.activation(self.end_proj(similarity)))

        bilinear_term = torch.einsum("bld,df,bmf->blm", start_repr, self.biaffine_weight, end_repr)
        start_bias = self.start_bias(start_repr)
        end_bias = self.end_bias(end_repr)
        logits = bilinear_term + start_bias + end_bias.transpose(1, 2) + self.biaffine_bias

        pair_mask = self._build_pair_mask(token_mask)
        logits = _mask_logits(logits, pair_mask)
        return logits

    def forward(self, similarity, token_mask, cross=False):
        if cross:
            num_query, num_video, length = similarity.shape
            flat_similarity = similarity.reshape(num_query * num_video, length)
            flat_mask = token_mask.unsqueeze(0).expand(num_query, -1, -1).reshape(num_query * num_video, length)
            flat_logits = self._forward_single(flat_similarity, flat_mask)
            return flat_logits.view(num_query, num_video, length, length)
        return self._forward_single(similarity, token_mask)
