import torch
from torch import nn
import torch.nn.functional as F
import math


"""经典时序预测模型集合。

本文件中的模型都接收一个历史窗口，并输出下一时刻的回归值。
常见输入形状为 ``[batch_size, sequence_length, feature_num]``；二维卷积
模型例外，需要额外的通道维，形状通常为
``[batch_size, 1, sequence_length, feature_num]``。

模型定位：
* Linear_Model：多层感知机基线，适合验证非时序特征组合能力。
* CNN_model：二维卷积基线，提取局部时间-特征模式。
* LSTM_model / GRU_model：循环神经网络，适合顺序依赖明显的短序列。
* CNN_LSTM_model：先提取局部模式，再建模时间依赖。
* Transformer：基于自注意力的序列模型，适合并行建模长距离依赖。
"""


class Linear_Model(nn.Module):
    """多层感知机（MLP）回归基线。

    原理：先将历史窗口展平，再通过两层 ReLU 全连接层学习输入特征
    与目标负荷之间的非线性关系。它不显式保留时间顺序，因此适合作为
    低复杂度基线或判断时序结构是否真正带来收益。

    输入：``[B, T, F]`` 或任意可展平为 ``input_size`` 的张量。
    输出：``[B, out_size]``。
    适用范围：样本量较小、窗口较短，或需要快速建立基线的回归任务。
    """

    def __init__(self, input_size, out_size, dropout=0.5):
        super(Linear_Model, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_size)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

    def forward(self, input):
        """展平历史窗口并输出回归结果。"""
        tem = self.flatten(input)
        tem = F.relu(self.fc1(tem))
        tem = F.relu(self.fc2(tem))
        # tem = self.dropout(tem)
        out = self.fc3(tem)

        return out


class CNN_model(nn.Module):
    """二维卷积负荷预测模型。

    原理：第一层卷积沿时间方向提取局部变化，第二层卷积沿特征方向
    融合相邻变量，随后使用池化压缩表示并通过全连接层回归。

    输入：``[B, 1, T, F]``。
    输出：``[B, 1]``。
    适用范围：时间窗口中存在局部模式、周期片段或变量间局部交互时。
    注意：当前全连接层固定为 ``64`` 个输入，输入窗口和卷积结构不能
    随意改变，否则需要同步调整 ``fc1`` 的输入维度。
    """

    def __init__(self):
        super(CNN_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 1), 1)
        self.conv2 = nn.Conv2d(32, 64, (1, 3), 1)
        self.pooling = nn.MaxPool2d((1, 4))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        """提取局部时空模式并预测下一时刻负荷。"""
        tem = F.relu(self.conv1(input))
        tem = F.relu(self.conv2(tem))
        tem = self.pooling(tem)
        tem = self.flat(tem)
        tem = F.relu(self.fc1(tem))
        # tem = self.dropout(tem)
        out = self.fc2(tem)

        return out


class LSTM_model(nn.Module):
    """长短期记忆网络（LSTM）回归模型。

    原理：LSTM 通过输入门、遗忘门和输出门控制信息流，能够缓解普通
    RNN 在长序列上的梯度消失问题。模型取最后一个时间步的隐藏状态，
    再通过全连接层输出预测值。

    输入：``[B, T, feature_num]``。
    输出：``[B, 1]``。
    适用范围：负荷受前几个小时或更长历史影响、且时间顺序重要的任务。
    """

    def __init__(self, feature_num, layer_num, nodes_num, dropout=0.3):
        super(LSTM_model, self).__init__()

        self.feature_num = feature_num
        self.layer_num = layer_num
        self.nodes_num = nodes_num

        self.lstm1 = nn.LSTM(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(nodes_num, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """沿时间维递推，并使用最后时间步的隐藏状态回归。"""
        tem, _ = self.lstm1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out

class GRU_model(nn.Module):
    """门控循环单元（GRU）回归模型。

    原理：GRU 用更新门和重置门控制历史信息，结构比 LSTM 更简单、
    参数更少，通常训练速度更快。其余预测头与 ``LSTM_model`` 类似。

    输入：``[B, T, feature_num]``。
    输出：``[B, 1]``。
    适用范围：希望以较低参数量建模时序依赖，或数据规模不足以支撑
    更复杂模型时。
    """

    def __init__(self, feature_num, layer_num, nodes_num, dropout=0.3):
        super(GRU_model, self).__init__()

        self.feature_num = feature_num
        self.layer_num = layer_num
        self.nodes_num = nodes_num

        self.gru1 = nn.GRU(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(nodes_num, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """沿时间维递推，并使用最后时间步的隐藏状态回归。"""
        tem, _ = self.gru1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out

class CNN_LSTM_model(nn.Module):
    """卷积-长短期记忆网络（CNN-LSTM）。

    原理：二维卷积先在每个时间步融合多个输入特征并提取局部表示，
    再把卷积输出重新排列为序列，交给 LSTM 建模跨时间依赖，最后用
    全连接层完成回归。

    输入：``[B, 1, T, feature_num]``。
    输出：``[B, 1]``。
    适用范围：既有局部变量交互，又有明显时间依赖的负荷、交通、能源
    和传感器序列预测任务。它是本项目中 CNN-LSTM 实验的核心模型。
    """

    def __init__(self, feature_num, conv_out=64, hidden_size=128, dropout=0.2):
        super(CNN_LSTM_model, self).__init__()

        self.feature_num = feature_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_out, kernel_size=(1, feature_num), stride=(1, 1)),
            nn.ReLU()
        )

        self.lstm3 = nn.LSTM(input_size=conv_out, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input):
        """先卷积融合特征，再用 LSTM 汇总时间信息。"""
        tem = self.conv1(input)
        tem = tem.squeeze(axis=-1)
        tem = tem.permute([0, 2, 1])
        tem, _ = self.lstm3(tem)
        tem = tem[:, -1, :]  # 取LSTM的最后一个时间步作为输出
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out


class Transformer(nn.Module):
    """基于 PyTorch Encoder 的 Transformer 回归模型。

    原理：多头自注意力允许序列中的任意时间步直接交互，前馈网络负责
    非线性变换。模型将编码后的完整序列展平，再通过线性层预测负荷。

    输入：当前实现期望 ``[T, B, feature_size]``，因为
    ``nn.TransformerEncoderLayer`` 默认使用 sequence-first 格式。
    输出：``[B, 1]``（当展平长度与 ``feature_size * 3`` 匹配时）。
    适用范围：需要捕获长距离时间依赖、并希望利用并行训练的序列任务。
    注意：``decoder`` 的输入维度固定为 ``feature_size * 3``，序列长度
    变化时需要同步调整该层。
    """

    def __init__(self, feature_size=7, num_layers=3, num_heads=1, dropout=0):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size * 3, 1)
        self.flatten = nn.Flatten()
        self.init_weights()

    def init_weights(self):
        """用较小的均匀分布初始化最终回归层。"""
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        """生成防止当前位置读取未来信息的上三角因果掩码。"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """执行 Transformer 编码并将序列表示映射到回归值。"""
        mask = self._generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, mask)
        output = self.flatten(output)
        output = self.decoder(output)
        return output


def main():
    x = torch.randn(32, 5, 4)
    model = Linear_Model(20, 1)
    # model = CNN_model()
    # model = LSTM_model(6, 1, 128)
    # model = CNN_LSTM_model(feature_num=4, conv_out=64, hidden_size=128)
    # model = GRU_model(feature_num=6, layer_num=1, nodes_num=128)
    # model = Transformer(feature_size=6, num_layers=3, num_heads=6, dropout=0)
    out = model(x)


    print(out.shape)


if __name__ == "__main__":
    main()
