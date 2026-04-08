import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding


class Model(nn.Module):
    """
    ExoTST: Exogenous-Aware Temporal Sequence Transformer
    Based on 2024 SOTA paper requirements for Net Ecosystem Exchange (NEE) prediction.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # Total variables (e.g., 9)
        self.exo_in = configs.enc_in - 1  # Exogenous variables (excluding target)
        
        # History Embedding (Inverted like iTransformer)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Future Exogenous Embedding (Standard temporal embedding)
        self.exo_embedding = DataEmbedding(self.exo_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder for History
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Aggregate Token (similar to [CLS] token in BERT/ViT)
        self.agg_token = nn.Parameter(torch.randn(1, 1, configs.d_model))
        
        # Cross-Temporal Modality Fusion (Aggregate-based Cross-Attention)
        self.fusion_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True),
            configs.d_model, configs.n_heads
        )

        # Projector
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 1. Historical Encoding (Inverted)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stds = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stds

        # enc_out: [B, D, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Add Aggregate Token for past context
        # B x (D + 1) x d_model
        agg = self.agg_token.repeat(enc_out.shape[0], 1, 1)
        enc_out = torch.cat([agg, enc_out], dim=1)
        
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # Extract the specialized aggregate token [B, 1, d_model]
        past_agg = enc_out[:, 0:1, :]
        hist_features = enc_out[:, 1:, :]

        # 2. Future Exogenous Encoding
        future_exo = x_dec[:, -self.pred_len:, :-1] # [B, S, D-1]
        future_exo_mark = x_mark_dec[:, -self.pred_len:, :]
        exo_out = self.exo_embedding(future_exo, future_exo_mark)

        # 3. Cross-Temporal Fusion (Official ExoTST logic: Aggregate Query)
        # Q: [B, 1, d_model], K/V: [B, S, d_model]
        query = past_agg
        key = value = exo_out
        
        # fused_agg: [B, 1, d_model]
        fused_agg, attns = self.fusion_attention(query, key, value, attn_mask=None)

        # Apply fused information back to variable tokens (Interaction)
        # Using a gated or additive fusion with projection
        fused_out = hist_features + fused_agg 
        
        # 4. Prediction
        # For NEE, the target variable is usually at a specific index. 
        # In this implementation, we allow all variable tokens to contribute to the final forecast.
        dec_out = self.projector(fused_out).permute(0, 2, 1)

        # De-Normalization
        dec_out = dec_out[:, :, :self.enc_in]
        dec_out = dec_out * (stds[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out # [B, S, D]
