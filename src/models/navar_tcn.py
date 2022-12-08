import torch.nn as nn
import torch


class TemporalBlock(nn.Module):
    """ copied from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
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
    """ copied from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script """
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


def receptive_field(kernel_size, layers):
    return (kernel_size - 1) * (2 ** (layers + 1) - 1) + 1


class NAVARTCN(nn.Module):
    def __init__(self, num_nodes, num_hidden, maxlags, hidden_layers=1, dropout=0):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            num_hidden: int
                Number of hidden units per layer
            maxlags: int
                Maximum number of time lags considered (K)
            hidden_layers: int
                Number of hidden layers
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVARTCN, self).__init__()

        channels = [16, 32]
        kernel_size = 3

        self.num_nodes = num_nodes
        self.lags_k = receptive_field(kernel_size, len(channels))

        self.tcn_list = nn.ModuleList([TemporalConvNet(1, channels, kernel_size=kernel_size, dropout=dropout)
                                      for _ in range(num_nodes)])
        self.fc_list = nn.ModuleList([nn.Linear(channels[-1], num_nodes)
                                      for _ in range(num_nodes)])
        self.biases = nn.Parameter(torch.ones(num_nodes, 1) * 0.0001)

    def forward(self, x):
        # x: (bs, num_nodes, time_steps)

        # we split the input into the components
        # x: num_nodes x (bs, 1, time_steps)
        x = x.split(1, dim=1)

        # x: num_nodes x (bs, time_steps, channels)
        x = [tcn(n).transpose(-2, -1) for n, tcn in zip(x, self.tcn_list)]

        # x: num_nodes x (bs, time_steps, num_nodes)
        x = [fc(n) for n, fc in zip(x, self.fc_list)]

        # x: (num_nodes, bs, time_steps, num_nodes)
        x = torch.stack(x)

        # x: (bs, num_nodes, num_nodes, time_steps)
        x = x.permute((1, 0, 3, 2))

        # predictions: (bs, num_nodes, time_steps)
        predictions = x.sum(dim=1) + self.biases

        # contributions: (bs * time_steps, num_nodes * num_nodes, 1)
        contributions = x.permute(0, 3, 1, 2).reshape(-1, self.num_nodes * self.num_nodes, 1)

        return predictions, contributions
