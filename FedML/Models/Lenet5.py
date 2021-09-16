# This is copied from https://github.com/buaabai/Ternary-Weights-Network
from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # [1, 28, 28]
        x = self.conv1(x)               # [32, 24, 24]
        x = F.relu(F.max_pool2d(x, 2))  # [32, 12, 12]
        x = self.conv2(x)               # [64, 8, 8]
        x = F.relu(F.max_pool2d(x, 2))  # [64, 4, 4]
        x = x.view(-1, 1024)            # [1024]
        x = F.relu(self.fc1(x))         # [512]
        x = self.fc2(x)                 # [10]
        return x


__all__ = ["LeNet5"]
