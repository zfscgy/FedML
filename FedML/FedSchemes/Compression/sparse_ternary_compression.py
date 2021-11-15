import numpy as np
import torch
from torch import nn

from typing import Callable, List
from dataclasses import dataclass

from FedML.Base.Utils import compute_lists_of_tensors, get_tensors, set_tensors, mean_tensors
from FedML.Base.config import GlobalConfig
from FedML.FedSchemes.Compression.base_compressor import BaseCompressor, \
    BaseCompressionClientOptions, BaseCompressionServerOptions, BaseCompressionClient, BaseCompressionServer
from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgServerOptions, FedAvgClient, FedAvgClientOptions



class STC(BaseCompressor):
    """
    From Paper 'Robust and Communication-Efficient Federated Learning from Non-IID Data' (aka STC, Sparse Ternary Compression)
    The sparsed ternary updates are uploaded to the server (using Golomb encoding to encode non-zero entries)
    The server broadcast the global model update to ALL clients.
    """
    def __init__(self, sparse_rate: float):
        self.sparse_rate = sparse_rate
        M = round(-1 / np.log2(1 - self.sparse_rate))
        b = round(np.log2(M))
        self.estimated_compression_ratio = self.sparse_rate / 2 * (b + 1 / (1 - np.power(1 - self.sparse_rate / 2, 2**b)))

    def compress(self, x: torch.Tensor):
        x_flat = torch.flatten(x)
        k = int(self.sparse_rate * int(x_flat.size()[0])) or 1
        topk_xs, _ = torch.topk(torch.abs(x_flat), k, sorted=True)
        topk_x = topk_xs[-1]

        big_indices = torch.abs(x) >= topk_x
        big_mean = torch.mean(torch.abs(x[big_indices]))

        ternarized_x = torch.zeros_like(x)
        ternarized_x[x <= -topk_x] = -big_mean
        ternarized_x[x >= topk_x] = big_mean

        # Use golomb coding to compress
        len = int(x_flat.size()[0])

        compressed_size = len * 2 * self.estimated_compression_ratio / 32 + 1  # The unit is float, not bit

        return ternarized_x, compressed_size


class STCServer(BaseCompressionServer):
    def __init__(self, get_model: Callable[[], nn.Module], options: BaseCompressionServerOptions):
        super(STCServer, self).__init__(get_model, options, residual_accumulation=True)


class STCClient(BaseCompressionClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: STCServer, options: BaseCompressionClientOptions):
        super(STCClient, self).__init__(get_model, server, options, residual_accumulation=True)



if __name__ == '__main__':
    a = torch.normal(0, 1, [1000])
    ternarized_a, compressed_size = STC(0.01).compress(a)
    print(compressed_size)
    print(ternarized_a)