"""带编码器和解码器的 Informer 模型。

Informer 面向长序列预测，使用 ProbSparse Attention 降低标准全注意力的
计算开销，并可通过蒸馏卷积逐层压缩序列。该变体保留完整的 Encoder-Decoder
结构，适合输出多个未来时间步，当前项目也可配置为单步预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.masking import TriangularCausalMask, ProbMask
from Informer_models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Informer_models.decoder import Decoder, DecoderLayer
from Informer_models.attn import FullAttention, ProbAttention, AttentionLayer
from Informer_models.embed import DataEmbedding


class Informer_1(nn.Module):
    """原始 Encoder-Decoder Informer。

    输入编码器历史序列 ``x_enc`` 和解码器输入 ``x_dec``，同时接收对应的
    时间特征 ``x_mark_enc``、``x_mark_dec``。输出为最后 ``out_len`` 个
    时间步的预测；当开启 ``output_attention`` 时，同时返回编码器注意力。
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=4, d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='relu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer_1, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """完成值/时间嵌入、编码器建模、解码器交叉注意力和投影。"""
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        dec_out = dec_out[:, -self.pred_len:, :].view(-1, 1)
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B * out_len, 1]


def main():
    x_enc = torch.rand(32, 3, 6)
    x_mark_enc = torch.zeros(32, 3, 4)
    x_dec = torch.rand(32, 4, 6)
    x_mark_dec = torch.zeros(32, 4, 4) #3+1
    model = Informer_1(enc_in=6, dec_in=6, c_out=1, seq_len=3, label_len=3, out_len=1)
    y = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(y.shape)


if __name__ == '__main__':
    main()