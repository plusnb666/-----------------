import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


"""时间卷积网络（Temporal Convolutional Network, TCN）。

TCN 使用因果卷积、扩张卷积和残差连接建模序列。相比循环网络，
卷积可以并行计算；扩张率逐层增大后，较少的层数也能覆盖更大的历史
感受野。当前实现面向单步回归预测，输入格式为 ``[B, T, F]``。
"""


class Chomp1d(nn.Module):
    """去除右侧多余 padding，保持因果卷积的输出长度和时序对齐。"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """裁剪最后 ``chomp_size`` 个时间位置。"""
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN 的基本残差块。

    结构为：权重归一化扩张卷积 → 裁剪 → ReLU → Dropout，重复两次，
    最后与输入（必要时先用 1x1 卷积变换通道数）相加。
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """用小方差正态分布初始化卷积权重。"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """计算残差块输出，保持时间维度与输入对齐。"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """多层时间卷积网络回归模型。

    输入：``[B, T, input_size]``。
    输出：``[B, out_size]``。
    适用范围：长时间窗口、局部模式明显且希望并行训练的时间序列。
    通过逐层增大的 dilation 覆盖不同尺度的历史信息，适合负荷、流量、
    传感器等具有局部周期或突变模式的任务。
    """

    def __init__(self, input_size, tcn_hidden_size, tcn_n_layers, tcn_dropout, out_size):
        super(TCN, self).__init__()
        self.input_size, self.tcn_hidden_size, self.tcn_n_layers, self.tcn_dropout, self.out_size = \
            input_size, \
            tcn_hidden_size, \
            tcn_n_layers, \
            tcn_dropout, \
            out_size
        num_channels = [self.tcn_hidden_size] * self.tcn_n_layers
        kernel_size = 2
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # one temporalBlock can be seen from fig1(b).
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=self.tcn_dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(self.tcn_hidden_size, self.out_size)

    def forward(self, x):
        """对序列做时间卷积，并使用最后一个时间位置进行回归。

        Args:
            x: ``[batch_size, seq_len, input_size]``。
        """
        # if self.args.importance:
        #     if not isinstance(x, torch.Tensor):
        #         x = torch.from_numpy(x)
        #     x = x.transpose(1, 2)
        #     x = x.to(torch.device("cuda"))
        x = x.transpose(1, 2)
        out = self.network(x)[:, :, -1:]  # 只保留最后一个时间位置
        out = self.fc(out.transpose(1, 2))
        out = out.view(-1, 1)
        return out


if __name__ == '__main__':
    import torch

    batch_size, seq_len, input_size = 32, 5, 4
    x = torch.randn((batch_size, seq_len, input_size))
    out_size = 1
    tcn_hidden_size = 128
    tcn_num_layers = 1
    drop_out = 0.2
    tcn = TCN(input_size, tcn_hidden_size, tcn_num_layers, drop_out, out_size)
    out = tcn(x)

    print(out.shape)
