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
        super(Linear_Model, self).__init__()  # 初始化父类 nn.Module

        self.input_size = input_size  # 保存展平后的输入特征数量
        self.out_size = out_size  # 保存模型最终输出的数量

        self.fc1 = nn.Linear(input_size, 128)  # 将输入特征映射为128个隐藏特征
        self.fc2 = nn.Linear(128, 64)  # 将128个隐藏特征压缩为64个隐藏特征
        self.fc3 = nn.Linear(64, out_size)  # 将64个隐藏特征映射为最终预测值
        self.dropout = nn.Dropout(dropout)  # 创建随机丢弃层以减少过拟合
        self.flatten = nn.Flatten()  # 创建展平层，将多维输入变成一维向量

    def forward(self, input):
        """展平历史窗口并输出回归结果。"""
        tem = self.flatten(input)  # 将输入形状[B,T,F]展平为[B,T*F]
        tem = F.relu(self.fc1(tem))  # 经过第一层全连接并使用ReLU激活函数
        tem = F.relu(self.fc2(tem))  # 经过第二层全连接并使用ReLU激活函数
        # tem = self.dropout(tem)  # 可选：随机丢弃部分特征以减少过拟合
        out = self.fc3(tem)  # 经过第三层全连接得到最终预测结果

        return out  # 返回模型的预测结果


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
        super(CNN_model, self).__init__()  # 初始化父类 nn.Module

        self.conv1 = nn.Conv2d(1, 32, (3, 1), 1)  # 使用32个卷积核沿时间方向提取局部模式
        self.conv2 = nn.Conv2d(32, 64, (1, 3), 1)  # 使用64个卷积核沿特征方向融合局部信息
        self.pooling = nn.MaxPool2d((1, 4))  # 沿特征方向进行最大池化并压缩数据
        self.flat = nn.Flatten()  # 将卷积和池化后的特征展平
        self.fc1 = nn.Linear(64, 32)  # 将64个卷积特征映射为32个隐藏特征
        self.fc2 = nn.Linear(32, 1)  # 将32个隐藏特征映射为一个预测值
        self.dropout = nn.Dropout(0.5)  # 创建随机丢弃层以减少过拟合

    def forward(self, input):
        """提取局部时空模式并预测下一时刻负荷。"""
        tem = F.relu(self.conv1(input))  # 第一层卷积提取局部时间模式并使用ReLU激活
        tem = F.relu(self.conv2(tem))  # 第二层卷积融合相邻特征并使用ReLU激活
        tem = self.pooling(tem)  # 保留局部区域中的最大响应并压缩特征尺寸
        tem = self.flat(tem)  # 将多维特征图展平为全连接层所需的一维特征
        tem = F.relu(self.fc1(tem))  # 通过第一层全连接并使用ReLU激活
        # tem = self.dropout(tem)  # 可选：随机丢弃部分特征以减少过拟合
        out = self.fc2(tem)  # 通过第二层全连接得到最终预测结果

        return out  # 返回模型预测值


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
        super(LSTM_model, self).__init__()  # 初始化父类 nn.Module

        self.feature_num = feature_num  # 保存每个时间步的输入特征数量
        self.layer_num = layer_num  # 保存LSTM堆叠的层数
        self.nodes_num = nodes_num  # 保存LSTM隐藏状态的特征数量

        self.lstm1 = nn.LSTM(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)  # 创建按批次优先处理的LSTM
        self.flat = nn.Flatten()  # 创建展平层以整理最后时间步的隐藏状态
        self.fc1 = nn.Linear(nodes_num, 64)  # 将LSTM隐藏特征映射为64个隐藏特征
        self.fc2 = nn.Linear(64, 1)  # 将64个隐藏特征映射为一个预测值
        self.dropout = nn.Dropout(dropout)  # 创建随机丢弃层以减少过拟合

    def forward(self, input):
        """沿时间维递推，并使用最后时间步的隐藏状态回归。"""
        tem, _ = self.lstm1(input)  # 按时间顺序处理输入并得到每个时间步的隐藏状态
        tem = tem[:, -1, :]  # 取每条样本最后一个时间步的隐藏状态
        tem = self.flat(tem)  # 将最后时间步的隐藏状态整理为二维特征
        # tem = self.dropout(tem)  # 可选：随机丢弃部分特征以减少过拟合
        tem = F.relu(self.fc1(tem))  # 通过第一层全连接并使用ReLU激活
        out = self.fc2(tem)  # 通过第二层全连接得到最终预测结果

        return out  # 返回模型预测值

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
        super(GRU_model, self).__init__()  # 初始化父类 nn.Module

        self.feature_num = feature_num  # 保存每个时间步的输入特征数量
        self.layer_num = layer_num  # 保存GRU堆叠的层数
        self.nodes_num = nodes_num  # 保存GRU隐藏状态的特征数量

        self.gru1 = nn.GRU(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)  # 创建按批次优先处理的GRU
        self.flat = nn.Flatten()  # 创建展平层以整理最后时间步的隐藏状态
        self.fc1 = nn.Linear(nodes_num, 64)  # 将GRU隐藏特征映射为64个隐藏特征
        self.fc2 = nn.Linear(64, 1)  # 将64个隐藏特征映射为一个预测值
        self.dropout = nn.Dropout(dropout)  # 创建随机丢弃层以减少过拟合

    def forward(self, input):
        """沿时间维递推，并使用最后时间步的隐藏状态回归。"""
        tem, _ = self.gru1(input)  # 按时间顺序处理输入并得到每个时间步的隐藏状态
        tem = tem[:, -1, :]  # 取每条样本最后一个时间步的隐藏状态
        tem = self.flat(tem)  # 将最后时间步的隐藏状态整理为二维特征
        # tem = self.dropout(tem)  # 可选：随机丢弃部分特征以减少过拟合
        tem = F.relu(self.fc1(tem))  # 通过第一层全连接并使用ReLU激活
        out = self.fc2(tem)  # 通过第二层全连接得到最终预测结果

        return out  # 返回模型预测值

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
        super(CNN_LSTM_model, self).__init__()  # 初始化父类 nn.Module

        self.feature_num = feature_num  # 保存每个时间步的输入特征数量

        self.conv1 = nn.Sequential(  # 创建卷积和激活组成的特征提取模块
            nn.Conv2d(1, conv_out, kernel_size=(1, feature_num), stride=(1, 1)),  # 使用卷积核融合每个时间步的全部特征
            nn.ReLU()  # 对卷积结果进行非线性变换
        )

        self.lstm3 = nn.LSTM(input_size=conv_out, hidden_size=hidden_size, batch_first=True, bidirectional=False)  # 使用LSTM学习卷积特征的时间依赖
        self.dropout = nn.Dropout(dropout)  # 创建随机丢弃层以减少过拟合
        self.fc1 = nn.Linear(128, 64)  # 将LSTM最后状态映射为64个隐藏特征
        self.fc2 = nn.Linear(64, 1)  # 将64个隐藏特征映射为一个预测值

    def forward(self, input):
        """先卷积融合特征，再用 LSTM 汇总时间信息。"""
        tem = self.conv1(input)  # 用卷积提取并融合每个时间步的局部特征
        tem = tem.squeeze(axis=-1)  # 删除卷积后长度为1的特征维度
        tem = tem.permute([0, 2, 1])  # 调整为LSTM需要的[B, T, 特征]形状
        tem, _ = self.lstm3(tem)  # 按时间顺序处理卷积特征并输出隐藏状态
        tem = tem[:, -1, :]  # 取LSTM的最后一个时间步作为序列总结
        # tem = self.dropout(tem)  # 可选：随机丢弃部分特征以减少过拟合
        tem = F.relu(self.fc1(tem))  # 通过第一层全连接并使用ReLU激活
        out = self.fc2(tem)  # 通过第二层全连接得到最终预测结果

        return out  # 返回模型预测值


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
        super(Transformer, self).__init__()  # 初始化父类 nn.Module
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)  # 创建一个Transformer编码层
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # 堆叠多个编码层形成编码器
        self.decoder = nn.Linear(feature_size * 3, 1)  # 将展平后的序列特征映射为一个预测值
        self.flatten = nn.Flatten()  # 创建展平层以整理编码器输出
        self.init_weights()  # 初始化最终回归层的权重和偏置

    def init_weights(self):
        """用较小的均匀分布初始化最终回归层。"""
        initrange = 0.1  # 设置回归层权重初始化范围
        self.decoder.bias.data.zero_()  # 将回归层偏置初始化为0
        self.decoder.weight.data.uniform_(-initrange, initrange)  # 在指定范围内均匀初始化回归层权重

    def _generate_square_subsequent_mask(self, sz):
        """生成防止当前位置读取未来信息的上三角因果掩码。"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # 创建禁止读取未来时间步的布尔掩码
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # 将禁止位置设为负无穷、允许位置设为0
        return mask  # 返回Transformer使用的因果掩码

    def forward(self, src):
        """执行 Transformer 编码并将序列表示映射到回归值。"""
        mask = self._generate_square_subsequent_mask(len(src))  # 根据输入序列长度生成因果掩码
        output = self.transformer_encoder(src, mask)  # 使用Transformer编码器提取序列依赖
        output = self.flatten(output)  # 将所有时间步的编码结果展平
        output = self.decoder(output)  # 将展平后的特征映射为预测值
        return output  # 返回模型预测值


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
