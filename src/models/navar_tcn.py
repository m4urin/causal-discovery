# copied from https://github.com/bartbussmann/NAVAR/blob/master/NAVAR.py
import math

import torch.nn as nn
import torch

from src.models.tcn import TCNClassifier


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
                Maximum number of time lags considered (2^K)
            hidden_layers: int
                Number of hidden layers
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVARTCN, self).__init__()

        self.num_nodes = num_nodes
        self.fc_list = nn.ModuleList([TCNClassifier(1, [8] * maxlags, dropout=dropout)])
        self.biases = nn.Parameter(torch.ones(1, num_nodes) * 0.0001)

    def forward(self, x):
        batch_size, number_of_nodes, time_series_length = x.shape
        contributions = torch.zeros((batch_size, self.num_nodes * self.num_nodes, time_series_length), device=x.device)

        # TODO: everything below
        # we split the input into the components
        x = x.split(1, dim=1)

        # then we apply the LSTM layers and calculate the contributions
        for node in range(self.num_nodes):
            model_input = torch.transpose(x[node], 1, 2)
            lstm = self.lstm_list[node]
            fc = self.fc_list[node]
            lstm_output, _ = lstm(model_input)
            contributions[:, node * self.num_nodes:(node + 1) * self.num_nodes, :] = fc(lstm_output).transpose(1, 2)

        contributions = contributions.view([batch_size, self.num_nodes, self.num_nodes, time_series_length])
        predictions = torch.sum(contributions, dim=1) + self.biases.transpose(0, 1)
        contributions = contributions.permute(0, 3, 1, 2)
        contributions = contributions.reshape(-1, self.num_nodes * self.num_nodes, 1)
        return predictions, contributions
