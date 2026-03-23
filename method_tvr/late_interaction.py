import torch
import torch.nn as nn
import torch.nn.functional as F


class EventTokenCompressor(nn.Module):
    """Lightweight event/anchor-aware query token compression for late interaction."""

    def __init__(
        self,
        keep_ratio=1.0,
        min_tokens=0,
        max_tokens=0,
        add_event_token=False,
        temperature=1.0,
        anchor_mode="boundary",
    ):
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.add_event_token = bool(add_event_token)
        self.temperature = float(temperature)
        self.anchor_mode = str(anchor_mode)

        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError("event_token_compression_keep_ratio must be in (0, 1]")
        if self.min_tokens < 0:
            raise ValueError("event_token_compression_min_tokens must be >= 0")
        if self.max_tokens < 0:
            raise ValueError("event_token_compression_max_tokens must be >= 0")
        if self.max_tokens > 0 and self.min_tokens > self.max_tokens:
            raise ValueError("event_token_compression_min_tokens cannot exceed max_tokens")
        if self.temperature <= 0:
            raise ValueError("event_token_compression_temperature must be > 0")
        if self.anchor_mode not in {"none", "boundary"}:
            raise ValueError("event_token_compression_anchor_mode must be one of {'none', 'boundary'}")

    def _select_indices_for_sample(self, sample_mask, sample_logits):
        valid_indices = torch.nonzero(sample_mask > 0, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return valid_indices

        valid_count = int(valid_indices.numel())
        target = int(round(valid_count * self.keep_ratio))
        target = max(1, target)
        if self.min_tokens > 0:
            target = max(target, self.min_tokens)
        if self.max_tokens > 0:
            target = min(target, self.max_tokens)
        target = min(target, valid_count)

        forced = []
        if self.anchor_mode == "boundary":
            forced = [int(valid_indices[0].item())]
            last_idx = int(valid_indices[-1].item())
            if last_idx != forced[0]:
                forced.append(last_idx)

        forced_set = set(forced)
        remaining_budget = max(0, target - len(forced))
        candidates = [int(idx.item()) for idx in valid_indices if int(idx.item()) not in forced_set]

        selected = list(forced)
        if remaining_budget > 0 and len(candidates) > 0:
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=sample_logits.device)
            candidate_logits = sample_logits[candidate_tensor]
            topk = min(remaining_budget, candidate_tensor.numel())
            top_rel = torch.topk(candidate_logits, k=topk, dim=0).indices
            selected.extend(candidate_tensor[top_rel].tolist())

        selected = sorted(set(selected))
        return torch.tensor(selected, dtype=torch.long, device=sample_mask.device)

    def forward(self, query_vectors, query_mask, token_logits):
        batch_size, _, dim = query_vectors.shape

        selected_indices_per_sample = []
        max_selected = 0
        for query_idx in range(batch_size):
            selected = self._select_indices_for_sample(query_mask[query_idx], token_logits[query_idx])
            selected_indices_per_sample.append(selected)
            max_selected = max(max_selected, int(selected.numel()))

        extra_slot = 1 if self.add_event_token else 0
        compressed_len = max_selected + extra_slot
        compressed_vectors = query_vectors.new_zeros((batch_size, compressed_len, dim))
        compressed_mask = query_mask.new_zeros((batch_size, compressed_len))
        compressed_logits = token_logits.new_full((batch_size, compressed_len), -1e4)
        compressed_indices = torch.full(
            (batch_size, compressed_len),
            -1,
            dtype=torch.long,
            device=query_vectors.device,
        )

        if compressed_len == 0:
            return compressed_vectors, compressed_mask, compressed_logits, compressed_indices

        for query_idx in range(batch_size):
            selected = selected_indices_per_sample[query_idx]
            if selected.numel() == 0:
                continue

            n_selected = int(selected.numel())
            selected_vectors = query_vectors[query_idx, selected]
            selected_logits = token_logits[query_idx, selected]

            compressed_vectors[query_idx, :n_selected] = selected_vectors
            compressed_mask[query_idx, :n_selected] = 1.0
            compressed_logits[query_idx, :n_selected] = selected_logits
            compressed_indices[query_idx, :n_selected] = selected

            if self.add_event_token:
                event_slot = n_selected
                if event_slot >= compressed_len:
                    continue
                event_weights = torch.softmax(selected_logits / self.temperature, dim=0)
                event_vector = torch.sum(selected_vectors * event_weights.unsqueeze(-1), dim=0)
                compressed_vectors[query_idx, event_slot] = F.normalize(event_vector, dim=0)
                compressed_mask[query_idx, event_slot] = 1.0
                compressed_logits[query_idx, event_slot] = selected_logits.max() + 1e-3
                compressed_indices[query_idx, event_slot] = -3  # sentinel for synthetic event token

        return compressed_vectors, compressed_mask, compressed_logits, compressed_indices


class LateInteractionRetriever(nn.Module):
    """ColBERT-style late interaction scorer with controlled multi-vector queries.

    Shapes:
        - query_vectors: (num_query, query_len, hidden_size) from query encoder tokens
        - query_mask: (num_query, query_len) token mask
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
        multi_vector_query_max_count=6,
        multi_vector_phrase_window=1,
        multi_vector_use_phrase_pooling=True,
        multi_vector_use_global_fallback=True,
        event_token_compression_enabled=False,
        event_token_compression_keep_ratio=1.0,
        event_token_compression_min_tokens=0,
        event_token_compression_max_tokens=0,
        event_token_compression_add_event_token=False,
        event_token_compression_temperature=1.0,
        event_token_compression_anchor_mode="boundary",
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
        self.multi_vector_query_max_count = int(multi_vector_query_max_count)
        self.multi_vector_phrase_window = int(multi_vector_phrase_window)
        self.multi_vector_use_phrase_pooling = bool(multi_vector_use_phrase_pooling)
        self.multi_vector_use_global_fallback = bool(multi_vector_use_global_fallback)
        self.event_token_compression_enabled = bool(event_token_compression_enabled)

        if self.score_reduction not in {"sum", "mean"}:
            raise ValueError("score_reduction must be 'sum' or 'mean'")
        if self.video_chunk_size <= 0:
            raise ValueError("video_chunk_size must be > 0")
        if self.multi_vector_query_max_count <= 0:
            raise ValueError("multi_vector_query_max_count must be > 0")
        if self.multi_vector_phrase_window < 0:
            raise ValueError("multi_vector_phrase_window must be >= 0")

        if self.use_projection:
            self.query_projection = nn.Linear(self.hidden_size, self.interaction_dim, bias=False)
            self.context_projection = nn.Linear(self.hidden_size, self.interaction_dim, bias=False)
        else:
            self.query_projection = nn.Identity()
            self.context_projection = nn.Identity()

        self.content_vector_scorer = nn.Linear(self.interaction_dim, 1, bias=False)
        if self.use_token_weight:
            self.query_vector_weight = nn.Linear(self.interaction_dim, 1, bias=False)
        else:
            self.query_vector_weight = None

        if self.event_token_compression_enabled:
            self.event_token_compressor = EventTokenCompressor(
                keep_ratio=event_token_compression_keep_ratio,
                min_tokens=event_token_compression_min_tokens,
                max_tokens=event_token_compression_max_tokens,
                add_event_token=event_token_compression_add_event_token,
                temperature=event_token_compression_temperature,
                anchor_mode=event_token_compression_anchor_mode,
            )
        else:
            self.event_token_compressor = None

    def _compute_content_logits(self, query_vectors, query_mask):
        logits = self.content_vector_scorer(query_vectors).squeeze(-1)
        return logits.masked_fill(query_mask <= 0, -1e4)

    def _build_multi_vector_query(self, query_vectors, query_mask, content_logits):
        batch_size, query_len, dim = query_vectors.shape
        extra_slots = 1 if self.multi_vector_use_global_fallback else 0
        max_slots = self.multi_vector_query_max_count + extra_slots

        multi_vectors = query_vectors.new_zeros((batch_size, max_slots, dim))
        multi_mask = query_vectors.new_zeros((batch_size, max_slots))
        multi_logits = query_vectors.new_full((batch_size, max_slots), -1e4)
        multi_indices = torch.full(
            (batch_size, max_slots),
            -1,
            dtype=torch.long,
            device=query_vectors.device,
        )

        for query_idx in range(batch_size):
            valid_indices = torch.nonzero(query_mask[query_idx] > 0, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                continue

            top_count = min(self.multi_vector_query_max_count, int(valid_indices.numel()))
            valid_logits = content_logits[query_idx, valid_indices]
            top_rel = torch.topk(valid_logits, k=top_count, dim=0).indices
            selected_indices = torch.sort(valid_indices[top_rel]).values

            slot = 0
            for token_index in selected_indices.tolist():
                if self.multi_vector_use_phrase_pooling and self.multi_vector_phrase_window > 0:
                    window_start = max(0, token_index - self.multi_vector_phrase_window)
                    window_end = min(query_len, token_index + self.multi_vector_phrase_window + 1)
                    window_mask = query_mask[query_idx, window_start:window_end] > 0
                    if torch.any(window_mask):
                        pooled_vector = query_vectors[query_idx, window_start:window_end][window_mask].mean(dim=0)
                    else:
                        pooled_vector = query_vectors[query_idx, token_index]
                    selected_vector = F.normalize(pooled_vector, dim=0)
                else:
                    selected_vector = query_vectors[query_idx, token_index]

                multi_vectors[query_idx, slot] = selected_vector
                multi_mask[query_idx, slot] = 1.0
                multi_logits[query_idx, slot] = content_logits[query_idx, token_index]
                multi_indices[query_idx, slot] = token_index
                slot += 1

            if self.multi_vector_use_global_fallback:
                valid_mask = query_mask[query_idx].float()
                global_vector = (query_vectors[query_idx] * valid_mask.unsqueeze(-1)).sum(dim=0)
                global_vector = global_vector / valid_mask.sum().clamp(min=1.0)
                multi_vectors[query_idx, slot] = F.normalize(global_vector, dim=0)
                multi_mask[query_idx, slot] = 1.0
                if slot > 0:
                    multi_logits[query_idx, slot] = multi_logits[query_idx, :slot].mean()
                else:
                    multi_logits[query_idx, slot] = 0.0
                multi_indices[query_idx, slot] = -2  # sentinel for global fallback vector

        return multi_vectors, multi_mask, multi_logits, multi_indices

    def _build_query_vector_weights(self, query_vectors, query_mask, query_logits):
        valid_mask = query_mask.float()
        if self.use_token_weight:
            logits = self.query_vector_weight(query_vectors).squeeze(-1)
        else:
            logits = query_logits
        logits = logits.masked_fill(valid_mask == 0, -1e4)
        weights = torch.softmax(logits, dim=-1) * valid_mask

        if self.token_weight_floor > 0:
            keep = (weights >= self.token_weight_floor).float() * valid_mask
            weights = weights * keep

        has_weight = weights.sum(dim=1, keepdim=True) > 0
        if not bool(torch.all(has_weight)):
            weights = torch.where(has_weight, weights, valid_mask)

        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return weights / denom

    def prepare_query_vectors(self, query_vectors, query_mask, return_details=False):
        projected = self.query_projection(query_vectors)
        projected = F.normalize(projected, dim=-1)
        content_logits = self._compute_content_logits(projected, query_mask)
        compression_info = None
        if self.event_token_compressor is not None:
            projected, query_mask, content_logits, compressed_indices = self.event_token_compressor(
                projected,
                query_mask,
                content_logits,
            )
            compression_info = {"compressed_token_indices": compressed_indices}

        multi_vectors, multi_mask, multi_logits, multi_indices = self._build_multi_vector_query(
            projected,
            query_mask,
            content_logits,
        )
        multi_weights = self._build_query_vector_weights(multi_vectors, multi_mask, multi_logits)
        if return_details:
            details = {
                "selected_indices": multi_indices,
                "selected_logits": multi_logits,
            }
            if compression_info is not None:
                details.update(compression_info)
            return multi_vectors, multi_mask, multi_weights, details
        return multi_vectors, multi_mask, multi_weights

    def prepare_context_vectors(self, context_vectors):
        projected = self.context_projection(context_vectors)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        query_vectors,
        query_mask,
        context_vectors,
        context_mask,
        context_is_prepared=False,
        return_diagnostics=False,
    ):
        if return_diagnostics:
            prepared_query, prepared_query_mask, query_vector_weights, query_info = self.prepare_query_vectors(
                query_vectors,
                query_mask,
                return_details=True,
            )
        else:
            prepared_query, prepared_query_mask, query_vector_weights = self.prepare_query_vectors(
                query_vectors,
                query_mask,
                return_details=False,
            )
            query_info = None
        if context_is_prepared:
            prepared_context = context_vectors
        else:
            prepared_context = self.prepare_context_vectors(context_vectors)
        outputs = self.score_prepared(
            prepared_query,
            prepared_query_mask,
            query_vector_weights,
            context_mask,
            prepared_context,
            return_contributions=return_diagnostics,
        )
        if not return_diagnostics:
            return outputs

        scores, vector_contributions = outputs
        diagnostics = {
            "query_vector_mask": prepared_query_mask,
            "query_vector_weights": query_vector_weights,
            "vector_contributions": vector_contributions,
        }
        if query_info is not None:
            diagnostics.update(query_info)
        return scores, diagnostics

    def score_prepared(
        self,
        query_vectors,
        query_vector_mask,
        query_vector_weights,
        context_mask,
        context_vectors,
        return_contributions=False,
    ):
        num_query = query_vectors.shape[0]
        num_video = context_vectors.shape[0]
        num_vector = query_vectors.shape[1]

        scores = query_vectors.new_zeros((num_query, num_video))
        vector_contributions = None
        if return_contributions:
            vector_contributions = query_vectors.new_zeros((num_query, num_vector, num_video))
        context_mask = context_mask.float()
        context_has_valid = context_mask.sum(dim=1) > 0
        denom = (query_vector_weights * query_vector_mask).sum(dim=1, keepdim=True).clamp(min=1e-6)

        for video_start in range(0, num_video, self.video_chunk_size):
            video_end = min(video_start + self.video_chunk_size, num_video)
            ctx_chunk = context_vectors[video_start:video_end]
            mask_chunk = context_mask[video_start:video_end]
            has_valid_chunk = context_has_valid[video_start:video_end]

            # (q, v, m, l): query batch, video chunk, query multi-vectors, video clips
            vector_sim = torch.einsum("qmd,vld->qvml", query_vectors, ctx_chunk)
            vector_sim = vector_sim.masked_fill(mask_chunk.unsqueeze(0).unsqueeze(2) == 0, -1e4)
            vector_max = vector_sim.max(dim=-1).values
            vector_max = vector_max.masked_fill(~has_valid_chunk.unsqueeze(0).unsqueeze(-1), 0.0)
            weighted_scores = vector_max * query_vector_weights.unsqueeze(1) * query_vector_mask.unsqueeze(1)
            chunk_scores = weighted_scores.sum(dim=-1)
            if self.score_reduction == "mean":
                chunk_scores = chunk_scores / denom
                if return_contributions:
                    weighted_scores = weighted_scores / denom.unsqueeze(1)

            scores[:, video_start:video_end] = chunk_scores
            if return_contributions:
                vector_contributions[:, :, video_start:video_end] = weighted_scores

        if return_contributions:
            return scores, vector_contributions
        return scores

    def score_selected_prepared(
        self,
        query_vectors,
        query_vector_mask,
        query_vector_weights,
        context_mask,
        context_vectors,
    ):
        """Score query-specific candidate sets.

        Args:
            query_vectors: (num_query, num_vector, dim)
            query_vector_mask: (num_query, num_vector)
            query_vector_weights: (num_query, num_vector)
            context_mask: (num_query, num_candidate, ctx_len)
            context_vectors: (num_query, num_candidate, ctx_len, dim)
        Returns:
            scores: (num_query, num_candidate)
        """
        context_mask = context_mask.float()
        context_has_valid = context_mask.sum(dim=-1) > 0
        denom = (query_vector_weights * query_vector_mask).sum(dim=1, keepdim=True).clamp(min=1e-6)

        vector_sim = torch.einsum("qmd,qkld->qkml", query_vectors, context_vectors)
        vector_sim = vector_sim.masked_fill(context_mask.unsqueeze(2) == 0, -1e4)
        vector_max = vector_sim.max(dim=-1).values
        vector_max = vector_max.masked_fill(~context_has_valid.unsqueeze(-1), 0.0)
        weighted_scores = vector_max * query_vector_weights.unsqueeze(1) * query_vector_mask.unsqueeze(1)
        scores = weighted_scores.sum(dim=-1)
        if self.score_reduction == "mean":
            scores = scores / denom
        return scores
