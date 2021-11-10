import numpy as np
import torch
from FedML.FedSchemes.Quantization.ternary_weight_network import BaseQuantization


class TernGrad(BaseQuantization):
    """
    From paper 'Terngrad'
    """
    def quantize(self, x: torch.Tensor):
        """
        Probabilistically set tensor to -1, 0, +1

        :param x:
        :return:
        """
        ternarized = torch.zeros_like(x)
        x_max = torch.max(x)
        zero_prob = torch.rand_like(x)
        ternarized[zero_prob < torch.abs(x) / x_max] = torch.sign(x[zero_prob < torch.abs(x) / x_max])
        return ternarized, np.prod(list(x.size())) * np.log2(3) / 32


if __name__ == '__main__':
    print(TernGrad().quantize(torch.normal(0, 1, [10])))