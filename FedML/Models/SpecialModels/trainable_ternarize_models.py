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
        grad_x = torch.zeros_like(grad_output)
        grad_x[torch.abs(x) <= delta] = grad_output[torch.abs(x) <= delta]
        grad_x[torch.abs(x) > delta] = grad_output[torch.abs(x) > delta] * w
        grad_w = torch.mean(grad_output[x > delta]) - torch.mean(grad_output[x < - delta])
        return grad_x, None, grad_w


trainable_ternarize = TrainableTernarize.apply


class TrainableTernaryBase(nn.Module):
    def get_ternarized_parameters(self) -> List[nn.Parameter]:
        raise NotImplementedError()

    def get_ternarized_values(self) -> List[torch.Tensor]:
        raise NotImplementedError()


class TrainableTernaryPara(TrainableTernaryBase):
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

    def get_ternarized_parameters(self) -> List[nn.Parameter]:
        return [self.full_precision_value]

    def get_ternarized_values(self) -> List[torch.Tensor]:
        return [self.forward()]


class TrainableTernarizedModel(TrainableTernaryBase):
    def get_ternarized_parameters(self) -> List[nn.Parameter]:
        values = []
        for m in self.children():
            if isinstance(m, TrainableTernaryBase):
                values += m.get_ternarized_parameters()
            else:
                values += m.parameters()
        return values

    def get_ternarized_values(self) -> List[torch.Tensor]:
        values = []
        for m in self.children():
            if isinstance(m, TrainableTernaryBase):
                values += m.get_ternarized_values()
            else:
                values += [p.data.clone() for p in m.parameters()]
        return values

    def get_compression_rate(self) -> float:
        raise NotImplementedError()

    def set_ternarized_values(self, tensors: List[torch.Tensor]):
        for p, t in zip(self.get_ternarized_parameters(), tensors):
            p.data = t.clone()



class TrainableTernaryLinear(nn.Linear, TrainableTernarizedModel):
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



class TrainableTernaryConv2d(nn.Conv2d, TrainableTernarizedModel):
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


class TrainableTernarizedLenet5(TrainableTernarizedModel):
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

    def get_compression_rate(self) -> float:
        para_list = list(self.parameters())
        original_floats = count_parameters(para_list) + len(para_list)  # Each parameter has a w para
        compressed_floats = count_parameters(para_list[:-2]) * np.log2(3) + len(para_list[:-2]) + \
                            count_parameters(para_list[-2:])
        return compressed_floats / original_floats


class TrainableTernarizedMnist2NN(TrainableTernarizedModel):
    def __init__(self, layer1_size: int, layer2_size: int):
        super(TrainableTernarizedMnist2NN, self).__init__()
        self.fc1 = TrainableTernaryLinear(784, layer1_size)
        self.fc2 = TrainableTernaryLinear(layer1_size, layer2_size)
        self.fc3 = TrainableTernaryLinear(layer2_size, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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