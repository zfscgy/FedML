from dataclasses import dataclass
import torch
from torch import nn
import numpy as np
from typing import List, Callable


from FedML.Base.base import Client, Server
from FedML.Base.config import GlobalConfig
from FedML.Base.Utils import get_tensors, compute_tensors, compute_lists_of_tensors, set_tensors, mean_tensors
from FedML.FedSchemes.fedavg import FedAvgServerOptions, FedAvgServer, FedAvgClientOptions, FedAvgClient


class BaseCompressor:
    def compress(self, x: torch.Tensor) -> [torch.Tensor, int]:
        """
        :param x:
        :return: Quantized tensor, compressed size
        """
        raise NotImplementedError()


def compress_by_batch(compressor: BaseCompressor, x: torch.Tensor, n_batch_dims=0):
    flat_to_shape = x.shape[n_batch_dims:]
    reshaped_x = x.reshape(-1, *flat_to_shape)
    compressed_x = torch.zeros_like(reshaped_x)
    compressed_size = 0
    for i in range(reshaped_x.shape[0]):
        compressed_x[i], compressed_size = compressor.compress(reshaped_x[i])
        compressed_size += compressed_size
    return compressed_x.view(*x.shape), compressed_size


def compress_tensor_list(compressor: BaseCompressor, tensors: List[torch.Tensor], excluded_indices: List[int] = None):
    if excluded_indices is None:
        excluded_indices = []
    compressed_size = 0
    compressed_tensors = []
    for i in range(len(tensors)):
        if i not in excluded_indices:
            ternarized_batch, compressed_size = compress_by_batch(compressor, tensors[i], n_batch_dims=1)
            compressed_tensors.append(ternarized_batch)
            compressed_size += compressed_size
            # log2(3) since only 3 values are needed
        else:
            compressed_tensors.append(tensors[i])
            compressed_size += np.prod(list(tensors[i].size()))
    return compressed_tensors, compressed_size

@dataclass
class BaseCompressionClientOptions(FedAvgClientOptions):
    compressor: BaseCompressor


class BaseCompressionClient(FedAvgClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: Server, options: BaseCompressionClientOptions,
                 residual_accumulation: bool):
        """
        :param get_model:
        :param server:
        :param options:
        :param compress_update:
        """
        super(BaseCompressionClient, self).__init__(get_model, server, options)
        self.residual_accumulation = residual_accumulation
        self.initial_model = None
        if residual_accumulation:
            self.residual_model = [torch.zeros_like(p) for p in self.local_model.parameters()]
        self.last_compressed_size = 0

    def update(self):
        self.initial_model = get_tensors(self.local_model, copy=True)
        return super(BaseCompressionClient, self).update()

    def get_compressed_weights(self) -> List[torch.Tensor]:
        update_tensors = compute_tensors(torch.sub, get_tensors(self.local_model), self.initial_model)

        if self.residual_accumulation:
            update_tensors = compute_tensors(torch.add, update_tensors, get_tensors(self.residual_model))

        compressed_tensors, compressed_size = compress_tensor_list(self.options.compressor, update_tensors)
        self.last_compressed_size = compressed_size

        if self.residual_accumulation:
            self.residual_model = compute_tensors(torch.sub, update_tensors, compressed_tensors)

        set_tensors(self.local_model, self.initial_model)

        return compressed_tensors


@dataclass
class BaseCompressionServerOptions(FedAvgServerOptions):
    compressor: BaseCompressor


class BaseCompressionServer(Server):
    def __init__(self, get_model: Callable[[], nn.Module], options: BaseCompressionServerOptions,
                 residual_accumulation: bool):
        super(BaseCompressionServer, self).__init__(get_model, options)
        self.residual_accumulation = residual_accumulation
        if residual_accumulation:
            self.residual_model = [0 for _ in self.global_model.parameters()]

    def update(self):
        clients = np.random.choice(self.clients, self.options.n_clients_per_round, replace=False)
        self.current_training_clients = clients

        client_new_weights = []
        for i, client in enumerate(clients):
            client.update()
            client_new_weights.append(client.get_compressed_weights())
            self.received_size += client.last_compressed_size

        updates = mean_tensors(client_new_weights)
        if self.residual_accumulation:
            updates = compute_tensors(torch.add, self.residual_model, updates)

        compressed_updates, compressed_size = compress_tensor_list(self.options.compressor, updates)
        set_tensors(self.global_model, compute_tensors(torch.add, get_tensors(self.global_model), compressed_updates))

        if self.residual_accumulation:
            self.residual_model = compute_tensors(torch.sub, updates, compressed_updates)

        for i, client in enumerate(self.clients):
            if i == 0 or not GlobalConfig.fast_mode:
                set_tensors(client.local_model, get_tensors(self.global_model, copy=True))
            self.sended_size += compressed_size
