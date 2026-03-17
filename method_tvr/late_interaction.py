import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteractionRetriever(nn.Module):
    """ColBERT-style late interaction scorer with mask-aware MaxSim aggregation.

    Shapes:
        - query_vectors: (num_query, query_len, hidden_size)
        - query_mask: (num_query, query_len)
        - context_vectors: (num_video, ctx_len, hidden_size)
        - context_mask: (num_video, ctx_len)
        - scores: (num_query, num_video)
    """

    def __init__(
        self,
        hidden_size,
        interaction_dim=0,
        use_projection=True,
        use_token_weight=False,
        token_weight_floor=0.0,
        score_reduction="mean",
        video_chunk_size=256,
    ):
        super().__init__()
        if interaction_dim <= 0:
            interaction_dim = hidden_size

        self.hidden_size = int(hidden_size)
        self.interaction_dim = int(interaction_dim)
        self.use_projection = bool(use_projection) or self.interaction_dim != self.hidden_size
        self.use_token_weight = bool(use_token_weight)
        self.token_weight_floor = float(token_weight_floor)
        self.score_reduction = str(score_reduction)
        self.video_chunk_size = int(video_chunk_size)

        if self.score_reduction not in {"sum", "mean"}:
            raise ValueError("score_reduction must be 'sum' or 'mean'")
        if self.video_chunk_size <= 0:
            raise ValueError("video_chunk_size must be > 0")

        if self.use_projection:
            self.query_projection = nn.Linear(self.hidden_size, self.interaction_dim, bias=False)
            self.context_projection = nn.Linear(self.hidden_size, self.interaction_dim, bias=False)
        else:
            self.query_projection = nn.Identity()
            self.context_projection = nn.Identity()

        if self.use_token_weight:
            self.query_token_weight = nn.Linear(self.interaction_dim, 1, bias=False)
        else:
            self.query_token_weight = None

    def prepare_query_vectors(self, query_vectors, query_mask):
        projected = self.query_projection(query_vectors)
        projected = F.normalize(projected, dim=-1)
        token_weights = self._build_query_token_weights(projected, query_mask)
        return projected, token_weights

    def prepare_context_vectors(self, context_vectors):
        projected = self.context_projection(context_vectors)
        return F.normalize(projected, dim=-1)

    def _build_query_token_weights(self, query_vectors, query_mask):
        valid_mask = query_mask.float()
        if not self.use_token_weight:
            return valid_mask

        logits = self.query_token_weight(query_vectors).squeeze(-1)
        logits = logits.masked_fill(valid_mask == 0, -1e4)
        weights = torch.softmax(logits, dim=-1) * valid_mask

        if self.token_weight_floor > 0:
            keep = (weights >= self.token_weight_floor).float() * valid_mask
            weights = weights * keep

        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return weights / denom

    def forward(
        self,
        query_vectors,
        query_mask,
        context_vectors,
        context_mask,
        context_is_prepared=False,
    ):
        prepared_query, query_token_weights = self.prepare_query_vectors(query_vectors, query_mask)
        if context_is_prepared:
            prepared_context = context_vectors
        else:
            prepared_context = self.prepare_context_vectors(context_vectors)
        return self.score_prepared(
            prepared_query,
            query_token_weights,
            context_mask,
            prepared_context,
        )

    def score_prepared(self, query_vectors, query_token_weights, context_mask, context_vectors):
        num_query = query_vectors.shape[0]
        num_video = context_vectors.shape[0]
        query_len = query_vectors.shape[1]

        scores = query_vectors.new_zeros((num_query, num_video))
        context_mask = context_mask.float()
        context_has_valid = context_mask.sum(dim=1) > 0

        for video_start in range(0, num_video, self.video_chunk_size):
            video_end = min(video_start + self.video_chunk_size, num_video)
            ctx_chunk = context_vectors[video_start:video_end]
            mask_chunk = context_mask[video_start:video_end]
            has_valid_chunk = context_has_valid[video_start:video_end]

            chunk_scores = query_vectors.new_zeros((num_query, video_end - video_start))
            for token_idx in range(query_len):
                token_weight = query_token_weights[:, token_idx]
                if torch.all(token_weight <= 0):
                    continue

                token_query = query_vectors[:, token_idx, :]
                token_sim = torch.einsum("qd,vld->qvl", token_query, ctx_chunk)
                token_sim = token_sim.masked_fill(mask_chunk.unsqueeze(0) == 0, -1e4)
                token_max = token_sim.max(dim=-1).values
                token_max = token_max.masked_fill(~has_valid_chunk.unsqueeze(0), 0.0)
                chunk_scores += token_max * token_weight.unsqueeze(1)

            scores[:, video_start:video_end] = chunk_scores

        if self.score_reduction == "mean":
            denom = query_token_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            scores = scores / denom
        return scores
