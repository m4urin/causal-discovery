import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import trange
import matplotlib.pyplot as plt

from src.models.navar_tcn import TemporalConvNet


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
        # (..., time_steps, channels)
        x = self.convnet(x).transpose(-2, -1)
        # (..., num_inputs, time_steps)
        return self.regressor(x).transpose(-2, -1)


if __name__ == '__main__':
    """ Small training example """
    warmup = 20
    T_train = 300
    T_test = 150
    data = torch.randn(warmup + T_train + T_test + 2, 3) / 4  # noise N(0, 0.5)
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

    optimizer = AdamW(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    train_loss, test_loss = [], []
    for epoch in trange(2000):
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
