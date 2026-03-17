import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from method_tvr.bimamba import BiMambaEncoderLayer
from method_tvr.contrastive import batch_video_query_loss
from method_tvr.late_interaction import LateInteractionRetriever
from method_tvr.model_components import (BertAttention, CrossAttentionLayer, LinearLayer,
                                         MILNCELoss, TrainablePositionalEncoding)
from method_tvr.query_decoder import QueryDecoder


class FusionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_to_video = CrossAttentionLayer(config)
        self.video_to_query = CrossAttentionLayer(config)

    @staticmethod
    def _build_cross_mask(query_mask, key_mask):
        return torch.einsum("bm,bn->bmn", query_mask, key_mask)

    def forward(self, query_states, video_states, query_mask, video_mask, return_attention=False):
        q2v_mask = self._build_cross_mask(query_mask, video_mask)
        v2q_mask = self._build_cross_mask(video_mask, query_mask)
        if return_attention:
            query_states, q2v_attn = self.query_to_video(query_states, video_states, q2v_mask, return_attention=True)
        else:
            query_states = self.query_to_video(query_states, video_states, q2v_mask)
            q2v_attn = None
        video_states = self.video_to_query(video_states, query_states, v2q_mask)
        return query_states, video_states, q2v_attn


class ReLoCLNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop,
        )
        self.ctx_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            hidden_size=config.hidden_size,
            dropout=config.input_drop,
        )

        self.query_input_proj = LinearLayer(
            config.query_input_size,
            config.hidden_size,
            layer_norm=True,
            dropout=config.input_drop,
            relu=True,
        )

        self.query_encoder = self._build_encoder_layer(config)
        self.query_encoder1 = copy.deepcopy(self.query_encoder)

        self.video_input_proj = LinearLayer(
            config.visual_input_size,
            config.hidden_size,
            layer_norm=True,
            dropout=config.input_drop,
            relu=True,
        )
        self.video_encoder1 = copy.deepcopy(self.query_encoder)
        self.video_encoder2 = copy.deepcopy(self.query_encoder)
        self.video_encoder3 = copy.deepcopy(self.query_encoder)
        self.video_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        # Single query head for the video branch.
        self.modular_vector_mapping = nn.Linear(config.hidden_size, 1, bias=False)

        conv_cfg = dict(
            in_channels=1,
            out_channels=1,
            kernel_size=config.conv_kernel_size,
            stride=config.conv_stride,
            padding=config.conv_kernel_size // 2,
            bias=False,
        )
        self.merged_st_predictor = nn.Conv1d(**conv_cfg)
        self.merged_ed_predictor = nn.Conv1d(**conv_cfg)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.nce_criterion = MILNCELoss(reduction="mean")

        self.use_generative_augmentation = getattr(config, "use_generative_augmentation", False)
        self.lm_weight = getattr(config, "lm_weight", 0.0)
        self.lm_pad_token_id = getattr(config, "lm_pad_token_id", 0)
        self.use_fusion_encoder = getattr(config, "use_fusion_encoder", False)
        self.fusion_num_layers = getattr(config, "fusion_num_layers", 2)
        self.retrieval_scorer = getattr(config, "retrieval_scorer", "single_vector")
        if self.retrieval_scorer not in {"single_vector", "late_interaction", "combined"}:
            raise ValueError("retrieval_scorer must be one of {'single_vector', 'late_interaction', 'combined'}")
        self.use_late_interaction = self.retrieval_scorer == "late_interaction"
        self.use_combined_retrieval = self.retrieval_scorer == "combined"
        self.use_late_component = self.retrieval_scorer in {"late_interaction", "combined"}
        self.combined_retrieval_alpha = float(getattr(config, "combined_retrieval_alpha", 0.5))
        self.combined_retrieval_normalize = str(getattr(config, "combined_retrieval_normalize", "zscore"))
        if self.combined_retrieval_normalize not in {"none", "zscore", "minmax"}:
            raise ValueError("combined_retrieval_normalize must be one of {'none', 'zscore', 'minmax'}")
        if not (0.0 <= self.combined_retrieval_alpha <= 1.0):
            raise ValueError("combined_retrieval_alpha must be in [0, 1]")

        if self.use_late_component:
            self.late_interaction_retriever = LateInteractionRetriever(
                hidden_size=config.hidden_size,
                interaction_dim=getattr(config, "late_interaction_dim", config.hidden_size),
                use_projection=getattr(config, "late_interaction_use_projection", True),
                use_token_weight=getattr(config, "late_interaction_use_token_weight", False),
                token_weight_floor=getattr(config, "late_interaction_token_weight_floor", 0.0),
                score_reduction=getattr(config, "late_interaction_score_reduction", "mean"),
                video_chunk_size=getattr(config, "late_interaction_video_chunk_size", 256),
            )
        else:
            self.late_interaction_retriever = None

        if self.use_generative_augmentation:
            self.query_decoder = QueryDecoder(
                vocab_size=config.lm_vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.lm_num_layers,
                num_heads=config.n_heads,
                dropout=config.drop,
                max_position_embeddings=config.max_desc_l,
            )

        if self.use_fusion_encoder:
            fusion_cfg = edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size * 4,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
                hidden_dropout_prob=config.drop,
            )
            self.fusion_layers = nn.ModuleList([FusionEncoderLayer(fusion_cfg) for _ in range(self.fusion_num_layers)])

        self.reset_parameters()

    @staticmethod
    def _build_encoder_layer(config):
        if config.backbone_type == "BiMamba":
            return BiMambaEncoderLayer(
                hidden_size=config.hidden_size,
                dropout=config.drop,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                fuse_mode=config.mamba_fuse_mode,
            )
        return BertAttention(
            edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
            )
        )

    @staticmethod
    def _format_attention_mask(mask, encoder_layer):
        if isinstance(encoder_layer, BiMambaEncoderLayer):
            return mask
        return mask.unsqueeze(1)

    def reset_parameters(self):
        """Initialize model weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        self.config.lw_st_ed = lw_st_ed

    def forward(
        self,
        query_feat,
        query_mask,
        video_feat,
        video_mask,
        st_ed_indices,
        match_labels,
        query_input_ids=None,
        query_attn_mask=None,
        return_aux=False,
    ):
        _, mid_x_video_feat, x_video_feat = self.encode_context(video_feat, video_mask, return_mid_output=True)
        outputs = self.get_pred_from_raw_query(
            query_feat,
            query_mask,
            x_video_feat,
            video_mask,
            cross=False,
            return_query_feats=True,
            return_encoded_query=self.use_fusion_encoder or self.use_late_component,
        )
        if self.use_fusion_encoder or self.use_late_component:
            video_query, query_context_scores, st_prob, ed_prob, encoded_query = outputs
        else:
            video_query, query_context_scores, st_prob, ed_prob = outputs
            encoded_query = None

        loss_fcl = 0
        if self.config.lw_fcl != 0:
            loss_fcl = batch_video_query_loss(mid_x_video_feat, video_query, match_labels, video_mask, measure="JSD")
            loss_fcl = self.config.lw_fcl * loss_fcl

        loss_vcl = 0
        if self.config.lw_vcl != 0:
            if self.retrieval_scorer != "single_vector":
                mid_video_q2ctx_scores = self._get_retrieval_scores(
                    video_query=video_query,
                    encoded_query=encoded_query,
                    query_mask=query_mask,
                    context_feat=mid_x_video_feat,
                    context_mask=video_mask,
                )
            else:
                mid_video_q2ctx_scores = self.get_unnormalized_video_level_scores(video_query, mid_x_video_feat, video_mask)
                mid_video_q2ctx_scores, _ = torch.max(mid_video_q2ctx_scores, dim=1)
            loss_vcl = self.nce_criterion(mid_video_q2ctx_scores)
            loss_vcl = self.config.lw_vcl * loss_vcl

        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = self.config.lw_st_ed * (loss_st + loss_ed)

        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)
            loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
            loss_neg_q = self.config.lw_neg_q * loss_neg_q

        loss_lm = 0
        if self.use_generative_augmentation and query_input_ids is not None and query_attn_mask is not None:
            fused_video_feat = x_video_feat
            if self.use_fusion_encoder:
                fused_video_feat, _ = self.fuse_query_video(
                    encoded_query,
                    query_mask,
                    x_video_feat,
                    video_mask,
                    return_attention=False,
                )
            loss_lm = self.compute_lm_loss(query_input_ids, query_attn_mask, fused_video_feat, video_mask)

        loss = loss_fcl + loss_vcl + loss_st_ed + loss_neg_ctx + loss_neg_q + self.lm_weight * loss_lm
        loss_dict = {
            "loss_st_ed": float(loss_st_ed),
            "loss_fcl": float(loss_fcl),
            "loss_vcl": float(loss_vcl),
            "loss_neg_ctx": float(loss_neg_ctx),
            "loss_neg_q": float(loss_neg_q),
            "loss_lm": float(loss_lm),
            "loss_overall": float(loss),
        }
        if return_aux:
            aux = {
                "query_context_scores": query_context_scores,
                "encoded_video_feat": x_video_feat,
            }
            return loss, loss_dict, aux
        return loss, loss_dict

    def encode_query(self, query_feat, query_mask, return_encoded_query=False):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder, self.query_pos_embed)
        encoded_query = self.query_encoder1(encoded_query, self._format_attention_mask(query_mask, self.query_encoder1))
        video_query = self.get_modularized_query(encoded_query, query_mask)
        if return_encoded_query:
            return video_query, encoded_query
        return video_query

    def encode_context(self, video_feat, video_mask, return_mid_output=False):
        encoded_video_feat = self.encode_input(video_feat, video_mask, self.video_input_proj, self.video_encoder1, self.ctx_pos_embed)
        mid_video_feat = self.video_encoder2(encoded_video_feat, self._format_attention_mask(video_mask, self.video_encoder2))
        x_video_feat = self.video_encoder3(mid_video_feat, self._format_attention_mask(video_mask, self.video_encoder3))
        if return_mid_output:
            return encoded_video_feat, mid_video_feat, x_video_feat
        return x_video_feat

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if isinstance(encoder_layer, BiMambaEncoderLayer):
            return encoder_layer(feat, mask)
        return encoder_layer(feat, mask.unsqueeze(1))

    def get_modularized_query(self, encoded_query, query_mask):
        modular_attention_scores = self.modular_vector_mapping(encoded_query)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_query = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)
        return modular_query[:, 0]

    @staticmethod
    def get_video_level_scores(modularized_query, context_feat, context_mask):
        modularized_query = F.normalize(modularized_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", modularized_query, context_feat)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)
        query_context_scores = mask_logits(query_context_scores, context_mask)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)
        return query_context_scores

    @staticmethod
    def get_unnormalized_video_level_scores(modularized_query, context_feat, context_mask):
        query_context_scores = torch.einsum("md,nld->mln", modularized_query, context_feat)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)
        query_context_scores = mask_logits(query_context_scores, context_mask)
        return query_context_scores

    def _normalize_retrieval_scores(self, scores):
        if self.combined_retrieval_normalize == "none":
            return scores
        if self.combined_retrieval_normalize == "zscore":
            mean = scores.mean(dim=1, keepdim=True)
            std = scores.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
            return (scores - mean) / std
        if self.combined_retrieval_normalize == "minmax":
            min_v = scores.min(dim=1, keepdim=True).values
            max_v = scores.max(dim=1, keepdim=True).values
            denom = (max_v - min_v).clamp(min=1e-6)
            return (scores - min_v) / denom
        raise ValueError("Unsupported combined_retrieval_normalize: {}".format(self.combined_retrieval_normalize))

    def encode_retrieval_context(self, context_feat):
        if self.use_late_component:
            return self.late_interaction_retriever.prepare_context_vectors(context_feat)
        return context_feat

    def _get_retrieval_scores(
        self,
        video_query,
        encoded_query,
        query_mask,
        context_feat,
        context_mask,
        late_context_feat=None,
        late_context_is_prepared=False,
    ):
        if self.use_late_component:
            if encoded_query is None:
                raise ValueError("encoded_query is required for retrieval_scorer={}".format(self.retrieval_scorer))
            if late_context_feat is None:
                late_context_feat = context_feat
                late_context_is_prepared = False
            late_scores = self.late_interaction_retriever(
                query_vectors=encoded_query,
                query_mask=query_mask,
                context_vectors=late_context_feat,
                context_mask=context_mask,
                context_is_prepared=late_context_is_prepared,
            )
            if self.use_late_interaction:
                return late_scores
            single_scores = self.get_video_level_scores(video_query, context_feat, context_mask)
            single_scores = self._normalize_retrieval_scores(single_scores)
            late_scores = self._normalize_retrieval_scores(late_scores)
            alpha = self.combined_retrieval_alpha
            return (1.0 - alpha) * single_scores + alpha * late_scores
        return self.get_video_level_scores(video_query, context_feat, context_mask)

    def score_queries_to_single_context(self, query_feat, query_mask, context_feat, context_mask):
        if context_feat.size(0) != 1:
            raise ValueError("context_feat must have batch size 1, got {}".format(context_feat.size(0)))
        if context_mask.size(0) != 1:
            raise ValueError("context_mask must have batch size 1, got {}".format(context_mask.size(0)))
        if self.use_late_component:
            video_query, encoded_query = self.encode_query(query_feat, query_mask, return_encoded_query=True)
            late_context_feat = self.encode_retrieval_context(context_feat)
            late_context_is_prepared = True
        else:
            video_query = self.encode_query(query_feat, query_mask)
            encoded_query = None
            late_context_feat = None
            late_context_is_prepared = False
        q2ctx_scores = self._get_retrieval_scores(
            video_query=video_query,
            encoded_query=encoded_query,
            query_mask=query_mask,
            context_feat=context_feat,
            context_mask=context_mask,
            late_context_feat=late_context_feat,
            late_context_is_prepared=late_context_is_prepared,
        )
        return q2ctx_scores.squeeze(1)

    def get_merged_score(self, video_query, video_feat, cross=False):
        video_query = self.video_query_linear(video_query)
        if cross:
            return torch.einsum("md,nld->mnl", video_query, video_feat)
        return torch.einsum("bd,bld->bl", video_query, video_feat)

    def get_merged_st_ed_prob(self, similarity, context_mask, cross=False):
        if cross:
            n_q, n_c, length = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, length)
            st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, length)
            ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, length)
        else:
            st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze(1)
            ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze(1)
        st_prob = mask_logits(st_prob, context_mask)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def get_pred_from_raw_query(
        self,
        query_feat,
        query_mask,
        video_feat,
        video_mask,
        retrieval_context_feat=None,
        cross=False,
        return_query_feats=False,
        return_encoded_query=False,
        return_similarity=False,
    ):
        need_encoded_query = return_encoded_query or return_similarity or self.use_late_component

        if need_encoded_query:
            video_query, encoded_query = self.encode_query(query_feat, query_mask, return_encoded_query=True)
        else:
            video_query = self.encode_query(query_feat, query_mask)
            encoded_query = None

        if retrieval_context_feat is None:
            retrieval_context_feat = video_feat
            retrieval_context_is_prepared = False
        else:
            retrieval_context_is_prepared = self.use_late_component

        q2ctx_scores = self._get_retrieval_scores(
            video_query=video_query,
            encoded_query=encoded_query,
            query_mask=query_mask,
            context_feat=video_feat,
            context_mask=video_mask,
            late_context_feat=retrieval_context_feat,
            late_context_is_prepared=retrieval_context_is_prepared,
        )
        similarity = self.get_merged_score(video_query, video_feat, cross=cross)
        st_prob, ed_prob = self.get_merged_st_ed_prob(similarity, video_mask, cross=cross)

        outputs = []
        if return_query_feats:
            outputs.append(video_query)
        outputs.extend([q2ctx_scores, st_prob, ed_prob])
        if return_encoded_query:
            outputs.append(encoded_query)
        if return_similarity:
            temporal_curve = self.get_temporal_curve(encoded_query, query_mask, video_feat, video_mask)
            outputs.append(temporal_curve)
        return tuple(outputs)

    def compute_lm_loss(self, input_ids, attention_mask, memory, memory_mask):
        if input_ids.size(1) < 2:
            return torch.tensor(0.0, device=input_ids.device)
        dec_in = input_ids[:, :-1]
        dec_attn = attention_mask[:, :-1]
        targets = input_ids[:, 1:]
        logits = self.query_decoder(dec_in, dec_attn, memory, memory_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.lm_pad_token_id)
        return loss

    @staticmethod
    def compute_temporal_curve_from_tokens(encoded_query, query_mask, context_feat, context_mask):
        token_sim = torch.einsum("bld,bmd->blm", encoded_query, context_feat)
        token_sim = token_sim * query_mask.unsqueeze(-1)
        denom = query_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        temporal_curve = token_sim.sum(dim=1) / denom
        return temporal_curve * context_mask

    @staticmethod
    def _attention_to_curve(attention_probs, query_mask):
        attn = attention_probs.mean(dim=1)
        token_mask = query_mask.unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp(min=1.0)
        return (attn * token_mask).sum(dim=1) / denom

    def fuse_query_video(self, encoded_query, query_mask, video_feat, video_mask, return_attention=False):
        if not self.use_fusion_encoder:
            return video_feat, None
        q = encoded_query
        v = video_feat
        curves = []
        for layer in self.fusion_layers:
            q, v, q2v_attn = layer(q, v, query_mask, video_mask, return_attention=return_attention)
            if return_attention:
                curves.append(self._attention_to_curve(q2v_attn, query_mask))
        if return_attention:
            temporal_curve = torch.stack(curves, dim=0).mean(dim=0)
            temporal_curve = temporal_curve * video_mask
            return v, temporal_curve
        return v, None

    def get_temporal_curve(self, encoded_query, query_mask, video_feat, video_mask):
        if self.use_fusion_encoder:
            _, temporal_curve = self.fuse_query_video(encoded_query, query_mask, video_feat, video_mask, return_attention=True)
            return temporal_curve
        return self.compute_temporal_curve_from_tokens(encoded_query, query_mask, video_feat, video_mask)

    def get_video_level_loss(self, query_context_scores):
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(
            query_context_scores.transpose(0, 1),
            query_context_scores_masked.transpose(0, 1),
        )
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz
        sampled_neg_score_indices = sorted_scores_indices[
            batch_indices,
            torch.randint(sample_min_idx, sample_max_idx, size=(bsz,)).to(scores.device),
        ]
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        if self.config.ranking_loss_type == "hinge":
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        if self.config.ranking_loss_type == "lse":
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        raise NotImplementedError("Only support 'hinge' and 'lse'")


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
