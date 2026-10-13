#2024.11.3自己搭的Trans，只用了编码器，效果尚可
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""项目自定义的 Encoder-only Transformer。

与 ``typical_models.py`` 中直接使用的 Transformer 不同，本实现先用一维
卷积完成值嵌入，再加入正弦位置编码，最后使用 Transformer Encoder 建模
时间步之间的全局依赖。模型取最后一个时间步进行单步回归。
"""


class PositionalEncoding(nn.Module):
    """固定正弦-余弦位置编码，不参与梯度更新。"""

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码的矩阵
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # (d_model/2)

        # 计算正弦和余弦值
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # 增加一个维度，变成 (1, max_len, d_model)

        self.register_buffer('pe', pe)  # 不需要梯度更新的参数

    def forward(self, x):
        """为 ``[B, T, d_model]`` 表示加入位置信息。"""
        seq_length = x.size(1)  # 获取序列长度
        x = x + self.pe[:, :seq_length, :]  # 添加位置编码
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """使用 circular padding 的一维卷积值嵌入层。

    输入为 ``[B, T, C]``，先转为 Conv1d 所需的 ``[B, C, T]``，再转回
    ``[B, T, d_model]``。循环 padding 可减少窗口边界处的信息断裂。
    """

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """将原始特征映射到 Transformer 隐藏维度。"""
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class my_Transformer(nn.Module):
    """自定义 Encoder-only Transformer 回归模型。

    输入：``[B, T, input_size]``。
    输出：``[B, out_size]``。
    适用范围：需要并行建模远距离依赖、同时希望通过卷积增强局部模式
    提取的中短期时间序列预测任务。
    """

    def __init__(self, input_size, trans_hidden_size, seq_len, trans_n_heads, trans_n_layers, out_size):
        super(my_Transformer, self).__init__()
        self.input_size = input_size
        self.trans_hidden_size = trans_hidden_size
        self.seq_len = seq_len
        self.n_trans_head = trans_n_heads
        self.trans_n_layers = trans_n_layers
        self.out_size = out_size

        # self.embedding_layer = nn.Linear(input_size, trans_hidden_size)
        self.embedding_layer = TokenEmbedding(input_size, trans_hidden_size)
        self.position_encoder = PositionalEncoding(d_model=trans_hidden_size, max_len=5)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.trans_hidden_size, nhead=self.n_trans_head, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=self.trans_n_layers)
        self.fc = nn.Linear(self.trans_hidden_size, self.out_size)

    def forward(self, x):
        """执行值嵌入、位置编码、Transformer 编码和最终回归。"""
        x = self.embedding_layer(x)
        x = self.position_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)[:, -1, :]
        out = self.fc(x)

        return out


def main():
    x = torch.rand(32, 5, 4)
    model = my_Transformer(input_size=4, trans_hidden_size=64, seq_len=5, trans_n_heads=4, trans_n_layers=1,
                           out_size=1)

    # # 打印整个模型的参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"整个模型的参数量: {total_params}")
    #
    # # 打印 TransformerEncoder 的参数量
    # transformer_params = sum(p.numel() for p in model.transformer.parameters())
    # print(f"TransformerEncoder 的参数量: {transformer_params}")

    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
