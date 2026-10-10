import torch
from torch import nn
import torch.nn.functional as F
import math


class Linear_Model(nn.Module):
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
        tem = self.flatten(input)
        tem = F.relu(self.fc1(tem))
        tem = F.relu(self.fc2(tem))
        # tem = self.dropout(tem)
        out = self.fc3(tem)

        return out


class CNN_model(nn.Module):
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
        tem = F.relu(self.conv1(input))
        tem = F.relu(self.conv2(tem))
        tem = self.pooling(tem)
        tem = self.flat(tem)
        tem = F.relu(self.fc1(tem))
        # tem = self.dropout(tem)
        out = self.fc2(tem)

        return out


class LSTM_model(nn.Module):
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
        tem, _ = self.lstm1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out

class GRU_model(nn.Module):
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
        tem, _ = self.gru1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out

class CNN_LSTM_model(nn.Module):
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
    def __init__(self, feature_size=7, num_layers=3, num_heads=1, dropout=0):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size * 3, 1)
        self.flatten = nn.Flatten()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
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
