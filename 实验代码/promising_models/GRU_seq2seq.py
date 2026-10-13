import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""带注意力解码器的 GRU Seq2Seq 模型。

该模型由编码器和解码器组成：编码器读取历史窗口并生成每个时间步的
隐藏表示，解码器根据上一时刻隐藏状态计算注意力权重，再逐步生成未来
预测。当前项目主要使用 ``pred_len=1`` 的单步预测模式。
"""


class GRUEncoder(nn.Module):
    """GRU 编码器，输出完整序列表示和最终隐藏状态。"""

    def __init__(self, gru_num_layers=1, input_feature_len=6, sequence_len=3, hidden_size=128, bidirectional=False):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = gru_num_layers
        self.gru_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=gru_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, input_seq):
        """编码输入序列。

        输入形状为 ``[B, T, F]``，返回编码序列 ``[B, T, H]`` 和隐藏状态。
        当前实现将初始隐藏状态固定创建在 ``cuda:0``，因此原始代码要求
        使用 CUDA 设备；在 CPU 环境运行前需要将该设备改为
        ``input_seq.device``。
        """
        ht = torch.zeros(self.num_layers * self.gru_directions, input_seq.size(0), self.hidden_size, device="cuda:0")
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.gru_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.gru_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        return gru_out, hidden.squeeze(0)


class GRUDecoder(nn.Module):
    """带 Bahdanau 风格加性注意力的 GRU 单元解码器。

    解码器把上一时刻隐藏状态和当前输入拼接起来，生成对编码序列的
    注意力分布，再用加权上下文更新 GRU 隐藏状态并输出下一步结果。
    """

    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_gru_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)

    def forward(self, encoder_output, prev_hidden, y):
        """根据当前输入和隐藏状态生成一个时间步的预测。"""
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保留最后一层的信息
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        gru_hidden = self.decoder_gru_cell(attention_combine, prev_hidden)
        output = self.out(gru_hidden)
        return output, gru_hidden


class GRU_seq2seq(nn.Module):
    """GRU 编码器-解码器序列到序列预测模型。

    输入：``xb`` 为 ``[B, T, input_size]``。
    输出：当前实现最终 reshape 为 ``[-1, 1]``。
    适用范围：多步预测、输出序列与输入序列长度不同，或需要注意力解释
    历史时间步贡献的任务。当前代码中的 teacher forcing 参数保留为接口，
    但 teacher forcing 分支已注释，实际前向过程使用模型自身上一时刻输出。
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers, pred_len, window_size, teacher_forcing=0.3):
        super().__init__()
        self.encoder = GRUEncoder(num_layers, input_size, window_size, hidden_size)
        self.decoder_cell = GRUDecoder(input_size, output_size, window_size, hidden_size)
        self.output_size = output_size
        self.input_size = input_size
        self.pred_len = pred_len
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size, output_size)

    def __call__(self, xb, yb=None):
        """编码历史序列并自回归生成 ``pred_len`` 个预测。"""
        input_seq = xb
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(self.pred_len, input_seq.size(0), self.input_size, device='cuda:0')
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)
        y_prev = input_seq[:, -1, :]
        for i in range(self.pred_len):
            # if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
            #     y_prev = yb[:, i].unsqueeze(1)
            gru_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = gru_output
            outputs[i, :, :] = gru_output
        outputs = outputs.permute(1, 0, 2)
        if self.output_size == 1:
            outputs = self.linear(outputs)
        outputs = outputs.view(-1, 1)
        return outputs


def main():
    device = torch.device('cuda:0')
    x = torch.rand(32, 5, 4).to(device)
    model = GRU_seq2seq(input_size=4, output_size=1, hidden_size=128, num_layers=1, pred_len=1, window_size=5).to(device)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
