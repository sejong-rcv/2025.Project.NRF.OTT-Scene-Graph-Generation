"""
Temporal module 학습 위해 tgt/ref frames와 batch 처리하도록 구현 
(기존 코드는 batch 축에 ref frames를 쌓아서 gpu 한장에 video 하나만 얹기 가능. batch_size=1로 강제 고정이었음)
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.position_encoding import PositionalEncoding3D, PositionalEncoding1D

# transformer
# encoder, decoder, interaction_decoder, temporal_query_layer1~3(TemporalQueryEncoderLayer)
class CDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 num_dec_layers_temporal=3, num_ref_frames=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None, matcher=None):
        super().__init__()
        # spatial encoder (instance)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # spatial decoder (instace)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # interaction decoder (for relation)
        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        
        # Temporal Interaction Module
        self.num_dec_layers_temporal = num_dec_layers_temporal
        self.query_temporal_interaction = args.query_temporal_interaction
        self.temporal_feature_encoder = args.temporal_feature_encoder
        self.instance_temporal_interaction = args.instance_temporal_interaction
        self.relation_temporal_interaction = args.relation_temporal_interaction
        self.seq_sort = args.seq_sort
        self.one_temp = args.one_temp # False. only use one temporal query interaction, not in a coarse-to-fine way

        if self.temporal_feature_encoder:
            self.temporal_encoder = copy.deepcopy(decoder_layer)
        
        if self.seq_sort:
                self.spatial_temporal_pe = PositionalEncoding3D(d_model) # TODO: whether need 3d positional encoding
                if self.query_temporal_interaction:
                    self.temporal_pe = PositionalEncoding1D(2 * d_model)
                else:
                    self.temporal_pe = PositionalEncoding1D(d_model)

        # Temporal Query Encoder Layers
        if self.query_temporal_interaction: # True 
            temporal_query_layer = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
            if num_dec_layers_temporal == 1:
                    # temporal context aggregation module layers
                    # 여기 사용
                    self.temporal_query_layer1 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                    self.temporal_query_layer2 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
                    self.temporal_query_layer3 = TemporalQueryEncoderLayer(d_model * 2, dim_feedforward, dropout, activation, nhead)
            else:
                self.temporal_query_decoder1 = QueryTransformerDecoder(temporal_query_layer, num_dec_layers_temporal)
                self.temporal_query_decoder2 = QueryTransformerDecoder(temporal_query_layer, num_dec_layers_temporal)
                self.temporal_query_decoder3 = QueryTransformerDecoder(temporal_query_layer, num_dec_layers_temporal)
            # self.ins_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
            # self.rel_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
        else:
            # Instance/Relation separated temporal interaction (Legacy support)
            if self.instance_temporal_interaction:
                self.ins_temporal_query_layer1 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.ins_temporal_query_layer2 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.ins_temporal_query_layer3 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.ins_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
            if self.relation_temporal_interaction:
                self.rel_temporal_query_layer1 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.rel_temporal_query_layer2 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.rel_temporal_query_layer3 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
                self.rel_temporal_interaction_decoder = TransformerDecoder(decoder_layer, num_dec_layers_temporal, nn.LayerNorm(d_model))
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_ref_frames = num_ref_frames
        self.use_matched_query = args.use_matched_query
        self.matcher = matcher
        self.num_queries = args.num_queries

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, class_embed_dict=None, targets=None, cur_idx=0):
        """
        src: [Num_Frames, B, C, H, W] 
             - DSGG_Multi_Diff에서 permute되어 들어옴.
        mask: [Num_Frames, B, H, W]
        pos_embed: [Num_Frames, B, C, H, W]
        query_embed: [Num_Queries, C]
        """                        
        # Class Embedding MLP for coarse-to-fine query interaction
        obj_class_embed = class_embed_dict['obj_class_embed']
        attn_class_embed = class_embed_dict['attn_class_embed']
        spatial_class_embed = class_embed_dict['spatial_class_embed']
        contacting_class_embed = class_embed_dict['contacting_class_embed']
        sub_bbox_embed = class_embed_dict['sub_bbox_embed']
        obj_bbox_embed = class_embed_dict['obj_bbox_embed']

        # src shape : [Num_Frames, Batch, C, H, W]        
        # 1. Prepare Inputs for Spatial Transformer (Flatten Batch & Frames)        
        assert src.dim() == 5, f"Expected 5D input [N, B, C, H, W], got {src.shape}"
        num_frames, bs, c, h, w = src.shape
        assert num_frames == self.num_ref_frames + 1, f"Expected {self.num_ref_frames+1} frames, got {num_frames}"
        
        # src shape : [Num_Frames, B, C, H, W]
       
        # Collapse Num_Frames (N) and Batch (B) into a single dimension -> Total_Batch = N * B
        # src: [N, B, C, H, W] -> [N*B, C, H, W]
        src_flat = src.reshape(num_frames * bs, c, h, w)
        mask_flat = mask.reshape(num_frames * bs, h, w)
        pos_flat = pos_embed.reshape(num_frames * bs, c, h, w)
        
        # Prepare for Transformer: [HW, Total_Batch, C]
        src_input = src_flat.flatten(2).permute(2, 0, 1) # [HW, N*B, C]
        pos_input = pos_flat.flatten(2).permute(2, 0, 1) # [HW, N*B, C]
        mask_input = mask_flat.flatten(1)                # [N*B, HW]
        
        # Expand Query Embedding: [Num_Q, C] -> [Num_Q, N*B, C]
        query_embed_input = query_embed.unsqueeze(1).repeat(1, num_frames * bs, 1)
        tgt = torch.zeros_like(query_embed_input)
        
        # 2. Spatial Encoder & Decoder (Instance Decoding) 
        # Run independent spatial encoding/decoding for ALL frames simultaneously
        memory = self.encoder(src_input, src_key_padding_mask=mask_input, pos=pos_input) # spatial encoder
        
        hopd_out, ins_attn_weight = self.decoder( # spatial decoder
            tgt, memory, 
            memory_key_padding_mask=mask_input, 
            pos=pos_input, 
            query_pos=query_embed_input
        ) # hopd_out: [Num_Layers, Num_Q, Total_Batch(N*B), C]
        
        
        # 3. Separate Current & Reference Frames (Batch-Aware Restore)
        # We focus on the last layer output for Temporal Aggregation
        # Transpose to [Total_Batch, Num_Q, C]
        last_ins_hs = hopd_out[-1].transpose(0, 1) # Last Layer: [Num_Q, Total_Batch, C] -> [Total_Batch, Num_Q, C]
        
        # Restore dimensions: [Total_Batch, Num_Q, C] -> [Num_Frames, B, Num_Q, C] -> [B, Num_Frames, Num_Q, C]
        # We must verify the reshaping order matches the flattening order.
        # Flattening : [N, B, ...] -> [N*B, ...] , So Unflattening must be: [N*B, ...] -> [N, B, ...]
        last_ins_hs = last_ins_hs.view(num_frames, bs, self.num_queries, self.d_model)
        last_ins_hs = last_ins_hs.transpose(0, 1) # Now [Batch, Num_Frames, Q, C]
                
        # Extract Current Frame Features
        # cur_idx is index in [0, ... N-1]
        cur_ins_hs = last_ins_hs[:, cur_idx, :, :] # [B, Num_Q, C]
        
        # Extract Reference Frame Features
        ref_ids = [i for i in range(num_frames) if i != cur_idx]
        ref_ins_hs = last_ins_hs[:, ref_ids, :, :] # [B, Num_Ref, Num_Q, C]
                
        # Flatten Reference Features: [B, Num_Ref * Num_Q, C]
        # This creates the candidate pool for attention
        ref_concat_hs = ref_ins_hs.flatten(1, 2)
        
        # Prepare Temporal Positional Encoding (if enabled)
        cur_hs_tpe, ref_hs_tpe = None, None
        if self.seq_sort:
            # Need to handle PE generation for batch
            # Assuming PE generates [N, Dim] -> expand to [B, N, Dim]
            if self.query_temporal_interaction:
                hs_tpe = self.temporal_pe((num_frames, 2 * self.d_model)).to(src.device)
            else:
                hs_tpe = self.temporal_pe((num_frames, self.d_model)).to(src.device)
            
            hs_tpe = hs_tpe.unsqueeze(0).repeat(bs, 1, 1) # [B, Num_Frames, Dim]
            
            # Expand to queries: [B, Num_Frames, Num_Q, Dim]
            hs_tpe = hs_tpe.unsqueeze(2).repeat(1, 1, self.num_queries, 1)

            cur_hs_tpe = hs_tpe[:, cur_idx, :, :] # [B, Num_Q, Dim]
            
            ref_hs_tpe = hs_tpe[:, ref_ids, :, :] # [B, Num_Ref, Num_Q, Dim]
            ref_hs_tpe = ref_hs_tpe.flatten(1, 2) # [B, Num_Ref*Num_Q, Dim]
        
        # 4. Temporal Interaction (Relation)
        # ---------------------------------------------------------
        # Note: In OED, Temporal Aggregation happens *after* an initial Relation Decoding pass?
        # Or does it use Instance Features to facilitate Relation Decoding?
        # Based on original code, it seems to perform Interaction Decoding first, then Temporal.
        
        # Run Interaction Decoder (Relation) on ALL frames
        # We need query_pos for Relation Decoder (which is Instance Feature)
        # hopd_out[-1]: [Num_Q, Total_Batch, C]
        interaction_query_embed = hopd_out[-1] # [Q, Total_Batch, C]
        interaction_tgt = torch.zeros_like(interaction_query_embed)

        interaction_decoder_out, rel_attn_weight = self.interaction_decoder(
            interaction_tgt, memory, 
            memory_key_padding_mask=mask_input,
            pos=pos_input, 
            query_pos=interaction_query_embed
        )
        # interaction_decoder_out: [Num_Layers, Num_Q, Total_Batch, C]
        
        # Prepare Relation Features for Temporal Aggregation
        last_rel_hs = interaction_decoder_out[-1].transpose(0, 1) # [Total_Batch, Num_Q, C]
        last_rel_hs = last_rel_hs.view(num_frames, bs, self.num_queries, self.d_model).transpose(0, 1) # [B, N, Q, C]

        cur_rel_hs = last_rel_hs[:, cur_idx, :, :] # [B, Q, C]
        ref_rel_hs = last_rel_hs[:, ref_ids, :, :] # [B, N_ref, Q, C]
        ref_concat_rel_hs = ref_rel_hs.flatten(1, 2) # [B, N_ref*Q, C]

        # Concatenate Instance + Relation Features
        # Current: [B, Num_Q, 2*C]
        cur_concat_hs = torch.cat([cur_ins_hs, cur_rel_hs], dim=-1)
        # Reference: [B, Num_Ref*Num_Q, 2*C]
        ref_total_hs = torch.cat([ref_concat_hs, ref_concat_rel_hs], dim=-1)
        
        # 5. Progressively Refined Module (PRM)        
        if self.query_temporal_interaction:
            # Calculate Scores to pick Top-K references
            # We treat (Ref Frame index, Query index) as a single candidate pool
            # Need to reshape/view properly to apply classifiers
            
            # Apply classifiers to Reference Features (Batch-wise)
            # Input: [B, Num_Ref*Num_Q, C]
            # Since classifiers are Linear layers, they work on last dim
            ref_obj_prob = obj_class_embed(ref_concat_hs).softmax(-1)[..., :-1].max(-1)[0]
            ref_attn_prob = attn_class_embed(ref_concat_rel_hs).softmax(-1)[..., :-1].max(-1)[0]
            ref_spatial_prob = spatial_class_embed(ref_concat_rel_hs).sigmoid().max(-1)[0]
            ref_contacting_prob = contacting_class_embed(ref_concat_rel_hs).sigmoid().max(-1)[0]

            # Overall Probability Score
            overall_probs = ref_obj_prob * ref_attn_prob * ref_spatial_prob * ref_contacting_prob # [B, Num_Ref*Num_Q]

            # Progressive Refinement Steps
            # Step 1: Top-K1
            k1 = 80 * self.num_ref_frames
            _, topk_idx1 = torch.topk(overall_probs, k1, dim=1) # [B, K1]
            
            # Gather features
            # index shape needs to be [B, K1, 2*C] for gather
            gather_idx1 = topk_idx1.unsqueeze(-1).repeat(1, 1, ref_total_hs.shape[-1])
            ref_input1 = torch.gather(ref_total_hs, 1, gather_idx1) 
            
            # Gather PE if needed
            ref_pe1 = None
            if self.seq_sort and ref_hs_tpe is not None:
                gather_pe_idx1 = topk_idx1.unsqueeze(-1).repeat(1, 1, ref_hs_tpe.shape[-1])
                ref_pe1 = torch.gather(ref_hs_tpe, 1, gather_pe_idx1)

            # Temporal Layer 1
            # Note: TemporalQueryLayer expects [Seq, Batch, Dim]
            # cur: [B, Q, 2C] -> [Q, B, 2C]
            # ref: [B, K1, 2C] -> [K1, B, 2C]
            if self.num_dec_layers_temporal == 1:
                cur_concat_hs = self.temporal_query_layer1(
                    cur_concat_hs.transpose(0, 1), 
                    ref_input1.transpose(0, 1),
                    query_pos=cur_hs_tpe.transpose(0, 1) if cur_hs_tpe is not None else None,
                    ref_query_pos=ref_pe1.transpose(0, 1) if ref_pe1 is not None else None
                ).transpose(0, 1)
            else:
                cur_concat_hs = self.temporal_query_decoder1(
                    cur_concat_hs.transpose(0, 1), 
                    ref_input1.transpose(0, 1),
                    query_pos=cur_hs_tpe.transpose(0, 1) if cur_hs_tpe is not None else None,
                    pos=ref_pe1.transpose(0, 1) if ref_pe1 is not None else None
                ).transpose(0, 1)

            # Step 2: Top-K2
            k2 = 50 * self.num_ref_frames
            _, topk_idx2 = torch.topk(overall_probs, k2, dim=1)
            gather_idx2 = topk_idx2.unsqueeze(-1).repeat(1, 1, ref_total_hs.shape[-1])
            ref_input2 = torch.gather(ref_total_hs, 1, gather_idx2)
            
            ref_pe2 = None
            if self.seq_sort and ref_hs_tpe is not None:
                gather_pe_idx2 = topk_idx2.unsqueeze(-1).repeat(1, 1, ref_hs_tpe.shape[-1])
                ref_pe2 = torch.gather(ref_hs_tpe, 1, gather_pe_idx2)

            if self.num_dec_layers_temporal == 1:
                cur_concat_hs = self.temporal_query_layer2(
                    cur_concat_hs.transpose(0, 1), 
                    ref_input2.transpose(0, 1),
                    query_pos=cur_hs_tpe.transpose(0, 1) if cur_hs_tpe is not None else None,
                    ref_query_pos=ref_pe2.transpose(0, 1) if ref_pe2 is not None else None
                ).transpose(0, 1)
            else:
                cur_concat_hs = self.temporal_query_decoder2(
                    cur_concat_hs.transpose(0, 1), 
                    ref_input2.transpose(0, 1),
                    query_pos=cur_hs_tpe.transpose(0, 1) if cur_hs_tpe is not None else None,
                    pos=ref_pe2.transpose(0, 1) if ref_pe2 is not None else None
                ).transpose(0, 1)
                
            # Step 3: Top-K3
            k3 = 30 * self.num_ref_frames
            _, topk_idx3 = torch.topk(overall_probs, k3, dim=1)
            gather_idx3 = topk_idx3.unsqueeze(-1).repeat(1, 1, ref_total_hs.shape[-1])
            ref_input3 = torch.gather(ref_total_hs, 1, gather_idx3)

            ref_pe3 = None
            if self.seq_sort and ref_hs_tpe is not None:
                gather_pe_idx3 = topk_idx3.unsqueeze(-1).repeat(1, 1, ref_hs_tpe.shape[-1])
                ref_pe3 = torch.gather(ref_hs_tpe, 1, gather_pe_idx3)
            
            if self.num_dec_layers_temporal == 1:
                cur_concat_hs = self.temporal_query_layer3(
                    cur_concat_hs.transpose(0, 1), 
                    ref_input3.transpose(0, 1),
                    query_pos=cur_hs_tpe.transpose(0, 1) if cur_hs_tpe is not None else None,
                    ref_query_pos=ref_pe3.transpose(0, 1) if ref_pe3 is not None else None
                ).transpose(0, 1)
            else:
                cur_concat_hs = self.temporal_query_decoder3(
                    cur_concat_hs.transpose(0, 1), 
                    ref_input3.transpose(0, 1),
                    query_pos=cur_hs_tpe.transpose(0, 1) if cur_hs_tpe is not None else None,
                    pos=ref_pe3.transpose(0, 1) if ref_pe3 is not None else None
                ).transpose(0, 1)

            # Split back to Instance and Relation Features
            final_ins_hs, final_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)
        
        else:
            # Fallback if no temporal interaction (just returns current frame features)
            final_ins_hs = cur_ins_hs
            final_rel_hs = cur_rel_hs

        # 6. Final Outputs
        # Note: We return [Batch, Num_Q, C] - only for the target frame!
        # The Memory output is only from the Target Frame's memory for any subsequent use (if any)
        
        # Need to reshape memory for return? Original code returned `memory` permuted.
        # Here we extract only the target frame's memory to save space/logic
        # memory: [HW, Total_Batch, C] -> [HW, N, B, C] -> [HW, B, C] (Target only)
        # But `engine.py` might not use memory output, or use it for auxiliary tasks.
        # Let's verify return values.
        
        # hopd_out for Aux loss: Needs to be Target Frame only?
        # Original code returned `hopd_out` and `interaction_decoder_out` which contained ALL frames.
        # But `final_ins_hs` is only for Target.
        # To be safe and consistent with "Single Frame Prediction", we should likely
        # return the full stack for Aux loss if we want supervision on Ref frames too, 
        # BUT OED usually only supervises the Target frame in the final output.
        # Given `dsgg_multi` returns `out` dictionary based on `final_ins_hs` (target only),
        # we don't need to return the full history.

        # Returning Attention Weights (Optional, requires reshaping if needed)
        # ins_attn_weight: [Total_Batch, Num_Q, H, W]. Slice for target.
        bs_idx = torch.arange(bs, device=src.device)
        # Target frame indices in Total_Batch: 0, N, 2N... + cur_idx
        target_batch_indices = bs_idx * num_frames + cur_idx
        
        target_ins_attn = ins_attn_weight[target_batch_indices] # [B, Num_Q, H, W]
        target_rel_attn = rel_attn_weight[target_batch_indices] # [B, Num_Q, H, W]

        # Return: final_ins, final_rel, attns...
        # memory return is usually [B, C, H, W]
        target_memory = memory[:, target_batch_indices, :].permute(1, 2, 0).view(bs, c, h, w)

        return final_ins_hs, final_rel_hs, target_ins_attn, target_rel_attn, target_memory
        
        
        
        
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # st_pos_embed = pos_embed
        # query_embed = query_embed.unsqueeze(1).repeat(1, n, 1) # [num_query, c] -> [num_query, 5, c]
        # mask = mask.flatten(1)

        # tgt = torch.zeros_like(query_embed)

        # memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) # pass spatial encoder

        # cur_memory = memory[:, cur_idx:cur_idx+1]
        # ref_ids = torch.as_tensor((list(range(cur_idx)) + list(range(cur_idx+1, self.num_ref_frames+1))), dtype=torch.long, device=memory.device)
        # assert len(ref_ids) == self.num_ref_frames
        # ref_memory = memory[:, ref_ids].transpose(0, 1).reshape(-1, 1, self.d_model)
        # # ref_memory = memory_list[:, :, ref_ids]

        # # Pair-wise Instance Decoder
        # hopd_out = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed) 
        # hopd_out = hopd_out.transpose(1, 2)
       
        # last_ins_hs = hopd_out[-1] # (bs, num_query, embed dim)
        # # last_ins_hs = torch.stack(torch.chunk(last_ins_hs, bs, dim=0), dim=0)
        # cur_ins_hs = last_ins_hs[cur_idx:cur_idx+1] # current frame queries
        # ref_ins_hs = last_ins_hs[ref_ids]           # reference frame queries. (num_ref_queries, 100, dim)
        # ref_ins_hs_list = torch.chunk(ref_ins_hs, self.num_ref_frames, dim=0) 
        # ref_ins_hs = torch.cat(ref_ins_hs_list, 1)
        
        # cur_hs_tpe, ref_hs_tpe = None, None
        # if self.seq_sort:
        #     if self.query_temporal_interaction:
        #         hs_tpe = self.temporal_pe((n, 2 * self.d_model)).to(last_ins_hs.device)
        #     else:
        #         hs_tpe = self.temporal_pe((n, self.d_model)).to(last_ins_hs.device)
        #     hs_tpe = hs_tpe[:, None].repeat(1, last_ins_hs.shape[1], 1)
        #     cur_hs_tpe = hs_tpe[cur_idx:cur_idx+1]
        #     ref_hs_tpe = hs_tpe[ref_ids]
        #     ref_hs_tpe_list = torch.chunk(ref_hs_tpe, self.num_ref_frames, dim=0)
        #     ref_hs_tpe = torch.cat(ref_hs_tpe_list, 1)
       
        # # Relation Decoder
        # interaction_query_embed = torch.cat([hopd_out[-1][:cur_idx], cur_ins_hs, hopd_out[-1][cur_idx+1:]])
        # interaction_query_embed = interaction_query_embed.permute(1, 0, 2)
        # interaction_tgt = torch.zeros_like(interaction_query_embed)
        # interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, memory_key_padding_mask=mask,
        #                                 pos=pos_embed, query_pos=interaction_query_embed) # 2d pe as no temopral interaction in spatial decoder
        # interaction_decoder_out = interaction_decoder_out.transpose(1, 2)
        # last_rel_hs = interaction_decoder_out[-1]
        # cur_rel_hs = last_rel_hs[cur_idx:cur_idx+1]
        # ref_rel_hs = last_rel_hs[ref_ids]
        # ref_rel_hs_list = torch.chunk(ref_rel_hs, self.num_ref_frames, dim=0)
        # ref_rel_hs = torch.cat(ref_rel_hs_list, 1)

        # if self.query_temporal_interaction: 
        #     ref_obj_prob = obj_class_embed(ref_ins_hs).softmax(-1)[..., :-1].max(-1)[0]
            
        #     ref_attn_prob = attn_class_embed(ref_rel_hs).softmax(-1)[..., :-1].max(-1)[0]
        #     ref_spatial_prob = spatial_class_embed(ref_rel_hs).sigmoid().max(-1)[0]
        #     ref_contacting_prob = contacting_class_embed(ref_rel_hs).sigmoid().max(-1)[0]

        #     cur_concat_hs = torch.cat([cur_ins_hs, cur_rel_hs], dim=-1) # current frame "instance + relation"
        #     ref_concat_hs = torch.cat([ref_ins_hs, ref_rel_hs], dim=-1) # reference frame  "instance + relation"
            
        #     # original scoreing method
        #     overall_probs = ref_obj_prob * ref_attn_prob * ref_spatial_prob * ref_contacting_prob # filtering score
            
        #     # # 교체안 1) scoring 방식 변경 : weighted log sum
        #     # eps = 1e-4
        #     # o = (ref_obj_prob.clamp_min(eps)).log()
        #     # a = (ref_attn_prob.clamp_min(eps)).log()
        #     # s = (ref_spatial_prob.clamp_min(eps)).log()
        #     # c = (ref_contacting_prob.clamp_min(eps)).log()
        
        #     # # 희소 predicate 보호: predicate 쪽 가중 ↑
        #     # alpha, beta, gamma, delta = 0.7, 1.0, 0.9, 0.9
        #     # overall_log = alpha * o + beta * a + gamma * s + delta * c
            
        #     # # topk는 값의 대소관계만 사용하므로 log값을 그대로 써도 무방
        #     # overall_probs = overall_log
    

        #     if self.one_temp: # only one temporal query interaction layer 
        #         if self.use_matched_query:
        #             ref_obj_logits = obj_class_embed(ref_ins_hs)
        #             ref_sub_bboxes = sub_bbox_embed(ref_ins_hs).sigmoid()
        #             ref_obj_bboxes = obj_bbox_embed(ref_ins_hs).sigmoid()
        #             ref_attn_logits = attn_class_embed(ref_rel_hs)
        #             ref_spatial_logits = spatial_class_embed(ref_rel_hs)
        #             ref_contacting_logits = contacting_class_embed(ref_rel_hs)
        #             matched_ref_index_list = []
        #             for i in range(self.num_ref_frames):
        #                 outputs_ref_i = {'pred_obj_logits': ref_obj_logits[:, i*100:(i+1)*100, :], 'pred_sub_boxes': ref_sub_bboxes[:, i*100:(i+1)*100, :],\
        #                                 'pred_obj_boxes': ref_obj_bboxes[:, i*100:(i+1)*100, :], 'pred_attn_logits': ref_attn_logits[:, i*100:(i+1)*100, :],\
        #                                 'pred_spatial_logits': ref_spatial_logits[:, i*100:(i+1)*100, :], 'pred_contacting_logits': ref_contacting_logits[:, i*100:(i+1)*100, :]}
        #                 targets_ref_i = [targets[i+1]]
        #                 indices_ref_i = self.matcher(outputs_ref_i, targets_ref_i)
        #                 for idx in indices_ref_i[0][0]:
        #                     matched_ref_index_list.append(idx.item() + i*100)
        #             sel_indexes = torch.as_tensor(matched_ref_index_list, dtype=torch.int64)[None, :]
        #             sel_indexes = sel_indexes.to(ref_rel_hs.device)
        #         if not self.use_matched_query:
        #             sel_num = 10 * self.num_ref_frames
        #             _, sel_indexes = torch.topk(overall_probs, sel_num, dim=1)
        #         ref_concat_hs_input = torch.gather(ref_concat_hs, 1, sel_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
        #         ref_hs_tpe_topk = ref_hs_tpe[:, sel_indexes[0]] if self.seq_sort else None
        #         if self.num_dec_layers_temporal == 1:
        #             cur_concat_hs = self.temporal_query_layer(cur_concat_hs, ref_concat_hs_input, cur_hs_tpe, ref_hs_tpe_topk)
        #         else:
        #             cur_concat_hs = self.temporal_query_decoder(cur_concat_hs, ref_concat_hs_input, cur_hs_tpe, ref_hs_tpe_topk)
        #         cur_concat_hs = cur_concat_hs.permute(1, 0, 2)
        #         cur_ins_hs, cur_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)
            
        #     else: # use 3 temporal query layers     
        #         _, topk_indexes = torch.topk(overall_probs, 80 * self.num_ref_frames, dim=1) # 80 per reference frames
        #         ref_concat_hs_input1 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1])) 
        #         ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #         if self.num_dec_layers_temporal == 1:
        #             cur_concat_hs = self.temporal_query_layer1(cur_concat_hs, ref_concat_hs_input1, cur_hs_tpe, ref_hs_tpe_topk)
        #         else:
        #             cur_concat_hs = self.temporal_query_decoder1(cur_concat_hs, ref_concat_hs_input1, cur_hs_tpe, ref_hs_tpe_topk)

        #         _, topk_indexes = torch.topk(overall_probs, 50 * self.num_ref_frames, dim=1) # 50 per reference frames
        #         ref_concat_hs_input2 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
        #         ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #         if self.num_dec_layers_temporal == 1:
        #             cur_concat_hs = self.temporal_query_layer2(cur_concat_hs, ref_concat_hs_input2, cur_hs_tpe, ref_hs_tpe_topk)
        #         else:
        #             cur_concat_hs = self.temporal_query_decoder2(cur_concat_hs, ref_concat_hs_input2, cur_hs_tpe, ref_hs_tpe_topk)

        #         _, topk_indexes = torch.topk(overall_probs, 30 * self.num_ref_frames, dim=1) # 30 per reference frames
        #         ref_concat_hs_input3 = torch.gather(ref_concat_hs, 1, topk_indexes.unsqueeze(-1).repeat(1, 1, ref_concat_hs.shape[-1]))
        #         ref_hs_tpe_topk = ref_hs_tpe[:, topk_indexes[0]] if self.seq_sort else None
        #         if self.num_dec_layers_temporal == 1:
        #             cur_concat_hs = self.temporal_query_layer3(cur_concat_hs, ref_concat_hs_input3, cur_hs_tpe, ref_hs_tpe_topk)
        #         else:
        #             cur_concat_hs = self.temporal_query_decoder3(cur_concat_hs, ref_concat_hs_input3, cur_hs_tpe, ref_hs_tpe_topk)

        #         cur_concat_hs = cur_concat_hs.permute(1, 0, 2)
        #         cur_ins_hs, cur_rel_hs = torch.split(cur_concat_hs, self.d_model, -1)

        #         # tgt = torch.zeros_like(cur_ins_hs)
        #         # cur_ins_hs = self.ins_temporal_interaction_decoder(tgt, cur_memory, memory_key_padding_mask=mask[cur_idx:cur_idx+1],
        #         #                         pos=pos_embed[:, cur_idx:cur_idx+1], query_pos=cur_ins_hs)
        #         # cur_rel_hs = self.rel_temporal_interaction_decoder(tgt, cur_memory, memory_key_padding_mask=mask[cur_idx:cur_idx+1],
        #         #                         pos=pos_embed[:, cur_idx:cur_idx+1], query_pos=cur_rel_hs)
        #     cur_ins_hs = cur_ins_hs.transpose(0, 1)
        #     cur_rel_hs = cur_rel_hs.transpose(0, 1)
        #     final_ins_hs = cur_ins_hs
        #     final_rel_hs = cur_rel_hs

        # return final_ins_hs, final_rel_hs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate # True

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        attn_weight = None
        
        for layer in self.layers:
            output, attn_weight = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos)
            # output = layer(output, memory, tgt_mask=tgt_mask,
            #                memory_mask=memory_mask,
            #                tgt_key_padding_mask=tgt_key_padding_mask,
            #                memory_key_padding_mask=memory_key_padding_mask,
            #                pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # return torch.stack(intermediate)
            return torch.stack(intermediate), attn_weight

        # return output
        return output, attn_weight


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        
        tgt2, attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weight

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        # multihead_attn에서 weight도 받음
        tgt2, attn_weight = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                            key=self.with_pos_embed(memory, pos),
        #                            value=memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weight

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



class QueryTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                query_pos: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos, ref_query_pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TemporalQueryEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()
        # self attention 
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention 
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # ffn 
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model) 

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, query, ref_query, query_pos=None, ref_query_pos=None):
        """
        Input Shapes: [Sequence_Length, Batch_Size, Embedding_Dim]
        """
        # Self Attention
        q = k = self.with_pos_embed(query, query_pos)
        # query는 이미 [Seq, Batch, Dim] 형태이므로 transpose 없이 바로 넣음
        tgt2 = self.self_attn(q, k, query)[0] 
        tgt = query + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Cross Attention 
        # Query: Target Frame (tgt)
        # Key/Value: Reference Frames (ref_query)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(ref_query, ref_query_pos),
            ref_query
        )[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # FFN
        tgt = self.forward_ffn(tgt)
        
        return tgt

# class TemporalQueryEncoderLayer(nn.Module):
#     def __init__(self, d_model = 256, d_ffn = 1024, dropout=0.1, activation="relu", n_heads = 8):
#         super().__init__()

#         # self attention 
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # cross attention 
#         self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         # ffn 
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model) 

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt
    
#     def forward(self, query, ref_query, query_pos=None, ref_query_pos=None):
#         # self.attention
#         q = k = self.with_pos_embed(query, query_pos)
#         tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), query.transpose(0, 1))[0].transpose(0, 1)
#         tgt = query + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)

#         # cross attention 
#         tgt2 = self.cross_attn(
#             self.with_pos_embed(tgt, query_pos).transpose(0, 1), 
#             self.with_pos_embed(ref_query, ref_query_pos).transpose(0, 1),
#             ref_query.transpose(0,1)
#         )[0].transpose(0,1)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # ffn
#         tgt = self.forward_ffn(tgt)

#         return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_cdn(args, matcher):
    return CDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers_hopd=args.dec_layers_hopd,
        num_dec_layers_interaction=args.dec_layers_interaction,
        num_dec_layers_temporal=args.dec_layers_temporal,
        num_ref_frames=args.num_ref_frames,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args,
        matcher=matcher
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
