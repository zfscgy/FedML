from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



from FedML.Base.Utils import count_parameters


class TrainableTernarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, delta: torch.Tensor, w: torch.Tensor):
        ctx.save_for_backward(x, delta, w)
        ternarized = torch.zeros_like(x)
        ternarized[x > delta] = w
        ternarized[x < - delta] = -w
        return ternarized

    @staticmethod
    def backward(ctx, grad_output):
        x, delta, w = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[torch.abs(x) > delta] *= w
        return grad_x, None, None


trainable_ternarize = TrainableTernarize.apply


class TrainableTernaryPara(nn.Module):
    def __init__(self, full_precision_para: torch.Tensor):
        super(TrainableTernaryPara, self).__init__()
        self.full_precision_value = nn.Parameter(full_precision_para)
        self.w = nn.Parameter(torch.mean(
            torch.abs(full_precision_para[torch.abs(full_precision_para) > 0.7 * torch.mean(torch.abs(full_precision_para))])))
        # The initialization is like normal ternary weight network

    def forward(self):
        delta = 0.05 * torch.max(torch.abs(self.full_precision_value))
        res = trainable_ternarize(self.full_precision_value, delta, self.w)
        return res


class TrainableTernaryBase(nn.Module):
    def get_ternarized_parameters(self) -> List[torch.Tensor]:
        raise NotImplementedError()


class TrainableTernaryLinear(nn.Linear, TrainableTernaryBase):
    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.ternarized_weight = TrainableTernaryPara(self.weight)
        if self.bias is not None:
            self.ternarized_bias = TrainableTernaryPara(self.bias)

    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.ternarized_weight(), self.ternarized_bias())
        else:
            return F.linear(x, self.ternarized_weight())

    def get_ternarized_parameters(self) -> List[torch.Tensor]:
        if self.bias is not None:
            return [self.ternarized_weight(), self.ternarized_bias()]
        else:
            return [self.ternarized_weight()]


class TrainableTernaryConv2d(nn.Conv2d, TrainableTernaryBase):
    def __init__(self, *args, **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        self.ternarized_weight = TrainableTernaryPara(self.weight)
        if self.bias is not None:
            self.ternarized_bias = TrainableTernaryPara(self.bias)

    def forward(self, x):
        if self.bias is not None:
            return super(TrainableTernaryConv2d, self)._conv_forward(x, self.ternarized_weight(), self.ternarized_bias())
        else:
            return super(TrainableTernaryConv2d, self)._conv_forward(x, self.ternarized_weight(), None)

    def get_ternarized_parameters(self) -> List[torch.Tensor]:
        if self.bias is not None:
            return [self.ternarized_weight(), self.ternarized_bias()]
        else:
            return [self.ternarized_weight()]


class TernarizedModel:
    def get_ternarized_parameters(self) -> List[torch.Tensor]:
        raise NotImplementedError()

    def get_ternarized_values(self) -> List[torch.Tensor]:
        raise NotImplementedError()

    def get_compression_rate(self) -> float:
        raise NotImplementedError()

    def set_ternarized_values(self, tensors: List[torch.Tensor]):
        for p, t in zip(self.get_ternarized_parameters(), tensors):
            p.data = t.clone()


class TrainableTernarizedLenet5(nn.Module, TernarizedModel):
    def __init__(self):
        super(TrainableTernarizedLenet5, self).__init__()
        self.conv1 = TrainableTernaryConv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = TrainableTernaryConv2d(32, 64, kernel_size=(5, 5))
        self.fc1 = TrainableTernaryLinear(1024, 512)
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

    def get_ternarized_parameters(self) -> List[nn.Parameter]:
        paras = self.conv1.get_ternarized_parameters() + self.conv2.get_ternarized_parameters() + \
                self.fc1.get_ternarized_parameters() + [p.data for p in self.fc2.parameters()]
        return paras

    def get_ternarized_values(self) -> List[torch.Tensor]:
        return [p.data.clone() for p in self.get_ternarized_parameters()]

    def get_compression_rate(self) -> float:
        para_list = list(self.parameters())
        original_floats = count_parameters(para_list) + len(para_list)  # Each parameter has a w para
        compressed_floats = count_parameters(para_list[:-2]) * np.log2(3) + len(para_list[:-2]) + \
                            count_parameters(para_list[-2:])
        return compressed_floats / original_floats


if __name__ == '__main__':
    para = torch.normal(0, 1, [10])
    ternary_para = TrainableTernaryPara(para)
    ternarized = ternary_para()
    loss = torch.sum(ternarized)
    loss.backward()
    pass