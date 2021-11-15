from dataclasses import dataclass
from typing import Callable, List
import torch
from torch import nn

from FedML.Base.base import Server
from FedML.FedSchemes.fedavg import FedAvgServerOptions, FedAvgClientOptions
from FedML.FedSchemes.Compression.base_compressor import BaseCompressor, \
    BaseCompressionServerOptions, BaseCompressionServer, BaseCompressionClientOptions, BaseCompressionClient


class SVDCompressor(BaseCompressor):
    def __init__(self, n_sigular_values: int = None, sparsity: float = None):
        self.n_sigular_values = n_sigular_values
        self.sparsity = sparsity

    def compress(self, x: torch.Tensor) -> [torch.Tensor, int]:
        if x.ndim == 1:
            x_2d = x.reshape([x.ndim // 2, -1])
        elif x.ndim == 2:
            x_2d = x
        else:
            x_2d = x.reshape([x.shape[0] * x.shape[1], -1])

        if self.n_sigular_values is None:
            self.n_sigular_values = int(x.shape[0] * x.shape[1] / (1 + x.shape[0] + x.shape[1]))

        u, s, v = torch.linalg.svd(x_2d)
        return (u[:, :self.n_sigular_values] * s[:self.n_sigular_values] @ v[:self.n_sigular_values, :])\
            .reshape(x.shape), \
            self.n_sigular_values * (x.shape[0] + x.shape[1] + 1)


@dataclass
class AtomoServerOptions(FedAvgServerOptions):
    sparsity: float


class AtomoServer(BaseCompressionServer):
    def __init__(self, get_model: Callable[[], nn.Module], options: AtomoServerOptions):
        base_options = BaseCompressionServerOptions(
            n_clients_per_round=options.n_clients_per_round,
            compressor=SVDCompressor(sparsity=options.sparsity)
        )
        super(AtomoServer, self).__init__(get_model, base_options, residual_accumulation=True)


@dataclass
class AtomoClientOptions(FedAvgClientOptions):
    sparsity: float


class AtomoClient(BaseCompressionClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: Server, options: AtomoServerOptions):
        arg_dict = options.__dict__
        sparsity = arg_dict.pop('sparsity')
        super(AtomoClient, self).__init__(get_model, server, BaseCompressionClientOptions(
            **arg_dict,
            compressor=SVDCompressor(sparsity=sparsity)
        ), residual_accumulation=True)


if __name__ == '__main__':
    a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    print(a @ a.transpose(0, 1))
    print(SVDCompressor(1).compress(a @ a.transpose(0, 1)))