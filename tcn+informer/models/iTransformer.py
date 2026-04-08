import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 记录原始特征维度
        self.output_attention = configs.output_attention
        
        # Embedding
        # Note: In iTransformer, we invert the dimensions: Time becomes Variate, and each Variate becomes a Token.
        # DataEmbedding_inverted expects (c_in, d_model) where c_in is seq_len.
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
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
        # Projector
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Inverted Embedding: [B, L, D] -> [B, D, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # Encoder: [B, D, d_model] -> [B, D, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projector: [B, D, d_model] -> [B, D, pred_len]
        dec_out = self.projector(enc_out).permute(0, 2, 1) # [B, pred_len, D]

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :self.enc_in]  # 返回原始特征维度的预测结果
