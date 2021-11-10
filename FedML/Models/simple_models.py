import torch
import torch.nn as nn


class MnistLogistic(nn.Module):
    def __init__(self):
        super(MnistLogistic, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)


class SimpleDNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SimpleDNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


class Mnist2NN(nn.Module):
    def __init__(self, layer1_size: int, layer2_size: int):
        super(Mnist2NN, self).__init__()
        self.linear1 = nn.Linear(784, layer1_size)
        self.linear2 = nn.Linear(layer1_size, layer2_size)
        self.linear3 = nn.Linear(layer2_size, 10)

    def forward(self, x):
        x1 = torch.relu(self.linear1(x))
        x2 = torch.relu(self.linear2(x1))
        y = self.linear3(x2)
        return y
