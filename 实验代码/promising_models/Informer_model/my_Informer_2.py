#2024.11.6 只带Encoder的Informer
import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.masking import TriangularCausalMask, ProbMask
from Informer_models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Informer_models.attn import FullAttention, ProbAttention, AttentionLayer
from Informer_models.embed import DataEmbedding


class Informer_2(nn.Module):
    def __init__(self, enc_in, d_hidden, c_out, seq_len, label_len, out_len,
                 factor=4, d_model=64, n_heads=4, e_layers=1, d_ff=128,
                 dropout=0.0,
                 attn='prob', embed='fixed', freq='h', activation='relu', distil=True):
        super(Informer_2, self).__init__()
        self.pred_len = out_len
        self.attn = attn

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
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

        self.projection1 = nn.Linear(d_model, d_hidden, bias=True)
        self.projection2 = nn.Linear(d_hidden, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection1(enc_out)
        out = self.projection2(enc_out)
        if self.pred_len != 1:
            out = out[:, -self.pred_len:, :]
        else:
            out = out[:, -self.pred_len, :]

        return out


def main():
    x_enc = torch.rand(32, 5, 4)
    x_mark_enc = torch.zeros(32, 5, 4)
    model = Informer_2(enc_in=4, d_hidden=64, c_out=1, seq_len=5, label_len=5, out_len=1,
                 factor=4, d_model=128, n_heads=8, e_layers=2, d_ff=256,
                 dropout=0.0)
    y = model(x_enc, x_mark_enc)

    print(y.shape)


if __name__ == '__main__':
    main()