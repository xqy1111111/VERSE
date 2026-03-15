import torch
import torch.nn as nn

from method_tvr.model_components import TrainablePositionalEncoding


def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class QueryDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout, max_position_embeddings):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = TrainablePositionalEncoding(max_position_embeddings, hidden_size, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, memory: torch.Tensor,
                memory_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T)
            attention_mask: (B, T)
            memory: (B, S, D)
            memory_mask: (B, S)
        """
        x = self.token_embed(input_ids)
        x = self.pos_embed(x)
        x = x.transpose(0, 1)  # (T, B, D)
        memory = memory.transpose(0, 1)  # (S, B, D)
        tgt_mask = _generate_causal_mask(x.size(0), x.device)
        tgt_key_padding_mask = attention_mask == 0
        memory_key_padding_mask = memory_mask == 0
        out = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = out.transpose(0, 1)  # (B, T, D)
        return self.lm_head(out)
