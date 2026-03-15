import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from method_tvr.model_components import (BertAttention, LinearLayer, BertSelfAttention,
                                         TrainablePositionalEncoding, CrossAttentionLayer)
from method_tvr.model_components import MILNCELoss
from method_tvr.contrastive import batch_video_query_loss
from method_tvr.bimamba import BiMambaEncoderLayer
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
        super(ReLoCLNet, self).__init__()
        self.config = config
        self.use_sub = "sub" in config.ctx_mode

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.ctx_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)

        self.query_encoder = self._build_encoder_layer(config)
        self.query_encoder1 = copy.deepcopy(self.query_encoder)

        cross_att_cfg = edict(hidden_size=config.hidden_size, num_attention_heads=config.n_heads,
                              attention_probs_dropout_prob=config.drop)
        # use_video
        self.video_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.video_encoder1 = copy.deepcopy(self.query_encoder)
        self.video_encoder2 = copy.deepcopy(self.query_encoder)
        self.video_encoder3 = copy.deepcopy(self.query_encoder)
        self.video_cross_att = BertSelfAttention(cross_att_cfg)
        self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
        self.video_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        # use_sub
        self.sub_input_proj = LinearLayer(config.sub_input_size, config.hidden_size, layer_norm=True,
                                          dropout=config.input_drop, relu=True)
        self.sub_encoder1 = copy.deepcopy(self.query_encoder)
        self.sub_encoder2 = copy.deepcopy(self.query_encoder)
        self.sub_encoder3 = copy.deepcopy(self.query_encoder)
        self.sub_cross_att = BertSelfAttention(cross_att_cfg)
        self.sub_cross_layernorm = nn.LayerNorm(config.hidden_size)
        self.sub_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.modular_vector_mapping = nn.Linear(in_features=config.hidden_size, out_features=2, bias=False)

        conv_cfg = dict(in_channels=1, out_channels=1, kernel_size=config.conv_kernel_size,
                        stride=config.conv_stride, padding=config.conv_kernel_size // 2, bias=False)
        self.merged_st_predictor = nn.Conv1d(**conv_cfg)
        self.merged_ed_predictor = nn.Conv1d(**conv_cfg)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.nce_criterion = MILNCELoss(reduction='mean')

        self.use_generative_augmentation = getattr(config, "use_generative_augmentation", False)
        self.lm_weight = getattr(config, "lm_weight", 0.0)
        self.lm_pad_token_id = getattr(config, "lm_pad_token_id", 0)
        self.use_fusion_encoder = getattr(config, "use_fusion_encoder", False)
        self.fusion_num_layers = getattr(config, "fusion_num_layers", 2)
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
                intermediate_size=config.hidden_size * 4,  # Fix: FFN intermediate layer should be 4x hidden_size
                num_attention_heads=config.n_heads,
                attention_probs_dropout_prob=config.drop,
                hidden_dropout_prob=config.drop,
            )
            self.fusion_layers = nn.ModuleList(
                [FusionEncoderLayer(fusion_cfg) for _ in range(self.fusion_num_layers)]
            )

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
        return BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                   hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                   attention_probs_dropout_prob=config.drop))

    @staticmethod
    def _format_attention_mask(mask, encoder_layer):
        if isinstance(encoder_layer, BiMambaEncoderLayer):
            return mask
        return mask.unsqueeze(1)

    def reset_parameters(self):
        """ Initialize the weights."""
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
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
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed

    def forward(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, st_ed_indices, match_labels,
                query_input_ids=None, query_attn_mask=None):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
            match_labels: (N, Lv), torch.LongTensor, matching labels for detecting foreground and background (not used)
        """
        video_feat, sub_feat, mid_x_video_feat, mid_x_sub_feat, x_video_feat, x_sub_feat = self.encode_context(
            video_feat, video_mask, sub_feat, sub_mask, return_mid_output=True)
        outputs = self.get_pred_from_raw_query(
            query_feat, query_mask, x_video_feat, video_mask, x_sub_feat, sub_mask, cross=False,
            return_query_feats=True, return_encoded_query=self.use_fusion_encoder)
        if self.use_fusion_encoder:
            video_query, sub_query, query_context_scores, st_prob, ed_prob, encoded_query = outputs
        else:
            video_query, sub_query, query_context_scores, st_prob, ed_prob = outputs
        # frame level contrastive learning loss (FrameCL)
        loss_fcl = 0
        if self.config.lw_fcl != 0:
            loss_fcl_vq = batch_video_query_loss(mid_x_video_feat, video_query, match_labels, video_mask, measure='JSD')
            if mid_x_sub_feat is not None and sub_query is not None:
                loss_fcl_sq = batch_video_query_loss(mid_x_sub_feat, sub_query, match_labels, sub_mask,
                                                     measure='JSD')
                loss_fcl = (loss_fcl_vq + loss_fcl_sq) / 2.0
            else:
                loss_fcl = loss_fcl_vq
            loss_fcl = self.config.lw_fcl * loss_fcl
        # video level contrastive learning loss (VideoCL)
        loss_vcl = 0
        if self.config.lw_vcl != 0:
            mid_video_q2ctx_scores = self.get_unnormalized_video_level_scores(video_query, mid_x_video_feat, video_mask)
            mid_video_q2ctx_scores, _ = torch.max(mid_video_q2ctx_scores, dim=1)
            if mid_x_sub_feat is not None and sub_query is not None:
                mid_sub_q2ctx_scores = self.get_unnormalized_video_level_scores(sub_query, mid_x_sub_feat, sub_mask)
                mid_sub_q2ctx_scores, _ = torch.max(mid_sub_q2ctx_scores, dim=1)
                mid_q2ctx_scores = (mid_video_q2ctx_scores + mid_sub_q2ctx_scores) / 2.0
            else:
                mid_q2ctx_scores = mid_video_q2ctx_scores
            loss_vcl = self.nce_criterion(mid_q2ctx_scores)
            loss_vcl = self.config.lw_vcl * loss_vcl
        # moment localization loss
        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed
            loss_st_ed = self.config.lw_st_ed * loss_st_ed
        # video level retrieval loss
        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)
            loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
            loss_neg_q = self.config.lw_neg_q * loss_neg_q
        # generative augmentation loss
        loss_lm = 0
        if self.use_generative_augmentation and query_input_ids is not None and query_attn_mask is not None:
            fused_video_feat = x_video_feat
            if self.use_fusion_encoder:
                fused_video_feat, _ = self.fuse_query_video(
                    encoded_query, query_mask, x_video_feat, video_mask, return_attention=False
                )
            loss_lm = self.compute_lm_loss(query_input_ids, query_attn_mask, fused_video_feat, video_mask)
        # sum loss
        loss = loss_fcl + loss_vcl + loss_st_ed + loss_neg_ctx + loss_neg_q + self.lm_weight * loss_lm
        return loss, {"loss_st_ed": float(loss_st_ed), "loss_fcl": float(loss_fcl), "loss_vcl": loss_vcl,
                      "loss_neg_ctx": float(loss_neg_ctx), "loss_neg_q": float(loss_neg_q),
                      "loss_lm": float(loss_lm), "loss_overall": float(loss)}

    def encode_query(self, query_feat, query_mask, return_encoded_query=False):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        encoded_query = self.query_encoder1(encoded_query, self._format_attention_mask(query_mask, self.query_encoder1))
        video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        if return_encoded_query:
            return video_query, sub_query, encoded_query
        return video_query, sub_query

    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask, return_mid_output=False):
        # encoding video features
        encoded_video_feat = self.encode_input(video_feat, video_mask, self.video_input_proj, self.video_encoder1,
                                               self.ctx_pos_embed)
        if not self.use_sub:
            x_encoded_video_feat_ = self.video_encoder2(encoded_video_feat,
                                                        self._format_attention_mask(video_mask, self.video_encoder2))
            x_encoded_video_feat = self.video_encoder3(
                x_encoded_video_feat_, self._format_attention_mask(video_mask, self.video_encoder3))
            if return_mid_output:
                return encoded_video_feat, None, x_encoded_video_feat_, None, x_encoded_video_feat, None
            return x_encoded_video_feat, None

        encoded_sub_feat = self.encode_input(sub_feat, sub_mask, self.sub_input_proj, self.sub_encoder1,
                                             self.ctx_pos_embed)
        x_encoded_video_feat = self.cross_context_encoder(encoded_video_feat, video_mask, encoded_sub_feat, sub_mask,
                                                          self.video_cross_att, self.video_cross_layernorm)
        x_encoded_video_feat_ = self.video_encoder2(
            x_encoded_video_feat, self._format_attention_mask(video_mask, self.video_encoder2))
        x_encoded_sub_feat = self.cross_context_encoder(encoded_sub_feat, sub_mask, encoded_video_feat, video_mask,
                                                        self.sub_cross_att, self.sub_cross_layernorm)
        x_encoded_sub_feat_ = self.sub_encoder2(
            x_encoded_sub_feat, self._format_attention_mask(sub_mask, self.sub_encoder2))
        x_encoded_video_feat = self.video_encoder3(
            x_encoded_video_feat_, self._format_attention_mask(video_mask, self.video_encoder3))
        x_encoded_sub_feat = self.sub_encoder3(
            x_encoded_sub_feat_, self._format_attention_mask(sub_mask, self.sub_encoder3))
        if return_mid_output:
            return (encoded_video_feat, encoded_sub_feat, x_encoded_video_feat_, x_encoded_sub_feat_,
                    x_encoded_video_feat, x_encoded_sub_feat)
        return x_encoded_video_feat, x_encoded_sub_feat

    @staticmethod
    def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer: cross attention layer
            norm_layer: layer norm layer
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        return residual_out

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if isinstance(encoder_layer, BiMambaEncoderLayer):
            return encoder_layer(feat, mask)
        mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask, return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        if return_modular_att:
            assert modular_queries.shape[1] == 2
            return modular_queries[:, 0], modular_queries[:, 1], modular_attention_scores
        else:
            assert modular_queries.shape[1] == 2
            return modular_queries[:, 0], modular_queries[:, 1]  # (N, D) * 2

    @staticmethod
    def get_video_level_scores(modularied_query, context_feat, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores, context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores

    @staticmethod
    def get_unnormalized_video_level_scores(modularied_query, context_feat, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores, context_mask)  # (N, L, N)
        return query_context_scores

    def get_merged_score(self, video_query, video_feat, sub_query, sub_feat, cross=False):
        video_query = self.video_query_linear(video_query)
        if sub_query is not None:
            sub_query = self.sub_query_linear(sub_query)
        if cross:
            video_similarity = torch.einsum("md,nld->mnl", video_query, video_feat)
            if sub_feat is not None and sub_query is not None:
                sub_similarity = torch.einsum("md,nld->mnl", sub_query, sub_feat)
                similarity = (video_similarity + sub_similarity) / 2
            else:
                similarity = video_similarity
        else:
            video_similarity = torch.einsum("bd,bld->bl", video_query, video_feat)  # (N, L)
            if sub_feat is not None and sub_query is not None:
                sub_similarity = torch.einsum("bd,bld->bl", sub_query, sub_feat)  # (N, L)
                similarity = (video_similarity + sub_similarity) / 2
            else:
                similarity = video_similarity
        return similarity

    def get_merged_st_ed_prob(self, similarity, context_mask, cross=False):
        if cross:
            n_q, n_c, length = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, length)
            st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
            ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
        else:
            st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
            ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def get_pred_from_raw_query(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, cross=False,
                                return_query_feats=False, return_encoded_query=False, return_similarity=False):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, D) or None
            video_mask: (N, Lv)
            sub_feat: (N, Lv, D) or None
            sub_mask: (N, Lv)
            cross:
            return_query_feats:
        """
        if return_similarity:
            return_encoded_query = True
        if return_encoded_query:
            video_query, sub_query, encoded_query = self.encode_query(query_feat, query_mask,
                                                                      return_encoded_query=True)
        else:
            video_query, sub_query = self.encode_query(query_feat, query_mask)
        # get video-level retrieval scores
        video_q2ctx_scores = self.get_video_level_scores(video_query, video_feat, video_mask)
        if sub_feat is not None and sub_query is not None:
            sub_q2ctx_scores = self.get_video_level_scores(sub_query, sub_feat, sub_mask)
            q2ctx_scores = (video_q2ctx_scores + sub_q2ctx_scores) / 2
        else:
            q2ctx_scores = video_q2ctx_scores
        # compute start and end probs
        similarity = self.get_merged_score(video_query, video_feat, sub_query, sub_feat, cross=cross)
        st_prob, ed_prob = self.get_merged_st_ed_prob(similarity, video_mask, cross=cross)
        outputs = []
        if return_query_feats:
            outputs.extend([video_query, sub_query])
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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=self.lm_pad_token_id)
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
            _, temporal_curve = self.fuse_query_video(
                encoded_query, query_mask, video_feat, video_mask, return_attention=True
            )
            return temporal_curve
        return self.compute_temporal_curve_from_tokens(encoded_query, query_mask, video_feat, video_mask)

    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1  # skip the masked positive
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz
        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == "lse":  # log[1 + exp(S_neg - S_pos)]
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
