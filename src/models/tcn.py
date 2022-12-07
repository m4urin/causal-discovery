# copied from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.optim import AdamW
from tqdm import trange


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, (1,)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNClassifier(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.convnet = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.regressor = nn.Sequential(*[
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Linear(num_channels[-1] // 2, num_inputs),
        ])

    def forward(self, x):
        """
        Args:
            x: Tensor of size (..., num_inputs, time_steps)
        Returns:
            Tensor of size (..., num_inputs, time_steps)
        """
        # x: (..., num_inputs, time_steps)
        n_dims = x.ndim
        # (..., channels, time_steps)
        x = self.convnet(x)
        # (..., time_steps, channels)
        x = x.permute((*range(x.ndim - 2), -1, -2))
        # (..., time_steps, num_inputs)
        x = self.regressor(x)
        # (..., num_inputs, time_steps)
        return x.permute((*range(x.ndim - 2), -1, -2))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    warmup = 20
    T_train = 300
    T_test = 150
    data = torch.randn(warmup + T_train + T_test + 2, 3) / 2  # noise N(0, 0.5)
    #data = torch.zeros(warmup + T_train + T_test + 2, 3)  # zeros
    for t in range(1, len(data)):
        data[t, 0] += torch.cos(data[t-1, 1]) + torch.tanh(data[t-1, 2])
        data[t, 1] += 0.35 * data[t - 1, 1] + data[t - 1, 2]
        data[t, 2] += torch.abs(0.5 * data[t - 1, 0]) + torch.sin(2 * data[t - 1, 1])
    plt.plot(data[warmup:warmup + 60])
    plt.show()
    plt.clf()

    DEVICE = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = data.permute((1, 0)).unsqueeze(0).detach().to(DEVICE)
    x_train = data[..., warmup + 0: warmup + T_train + 0]
    y_train = data[..., warmup + 1: warmup + T_train + 1]
    x_test = data[..., warmup + T_train + 1: warmup + T_train + T_test + 1]
    y_test = data[..., warmup + T_train + 2: warmup + T_train + T_test + 2]

    model = TCNClassifier(3, [16, 32, 32]).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=0.00009)
    criterion = nn.MSELoss()

    train_loss, test_loss = [], []
    for epoch in trange(2200):
        model.train()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        train_loss.append(loss / T_train)

        with torch.no_grad():
            model.eval()
            loss = criterion(model(x_test), y_test).item()
            test_loss.append(loss / T_test)

    plt.title('Loss')
    plt.plot(train_loss[10:], label='train loss')
    plt.plot(test_loss[10:], label='test loss')
    plt.legend()
    plt.show()
    plt.clf()

    model.eval()
    with torch.no_grad():
        for name, _x, _y in [('Train', x_train, y_train), ('Test', x_test, y_test)]:
            for i in range(3):
                plt.title(f'{name}: X_{i+1}')
                plt.plot(model(_x)[0, i, 10:70], label='pred')
                plt.plot(_y[0, i, 10:70], label='true')
                plt.legend()
                plt.show()
                plt.clf()
