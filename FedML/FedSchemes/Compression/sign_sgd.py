from dataclasses import dataclass
import numpy as np
import torch
from torch.optim import SGD

from typing import Callable

from FedML.Base.config import GlobalConfig
from FedML.Base.Utils import Convert, set_tensors, get_tensors, compute_tensors, mean_tensors
from FedML.Base.base import Server
from FedML.FedSchemes.fedavg import FedAvgServerOptions, FedAvgClientOptions
from FedML.FedSchemes.Compression.base_compressor import compress_tensor_list, BaseCompressor, \
    BaseCompressionClientOptions, BaseCompressionServerOptions, BaseCompressionClient, BaseCompressionServer


class SignCompressor(BaseCompressor):

    def compress(self, x: torch.Tensor) -> [torch.Tensor, int]:
        compressed_x = torch.sign(x)
        compressed_x[compressed_x == 0] = \
            Convert.to_tensor(torch.randint(0, 2, compressed_x[compressed_x == 0].shape).float()) * 2 - 1
        return compressed_x, x.flatten().shape[0] / 32

@dataclass
class SignSGDServerOptions(FedAvgServerOptions):
    learning_rate: float
    momentum: float


class SignSGDServer(Server):
    def __init__(self, get_model: Callable, options: SignSGDServerOptions):
        super(SignSGDServer, self).__init__(get_model, options)
        self.compressor = SignCompressor()
        self.optimizer = SGD(self.global_model.parameters(), options.learning_rate, options.momentum)

    def update(self):
        clients = np.random.choice(self.clients, self.options.n_clients_per_round, replace=False)
        self.current_training_clients = clients

        client_new_weights = []
        for i, client in enumerate(clients):
            client.update()
            client_new_weights.append(client.get_compressed_weights())
            self.received_size += client.last_compressed_size

        updates = mean_tensors(client_new_weights)
        compressed_updates, compressed_size = compress_tensor_list(self.compressor, updates)

        self.global_model.zero_grad()
        for para, update in zip(self.global_model.parameters(), compressed_updates):
            para.grad = -update
        self.optimizer.step()

        for i, client in enumerate(self.clients):
            if i == 0 or not GlobalConfig.fast_mode:
                set_tensors(client.local_model, get_tensors(self.global_model), copy=True)
            self.sended_size += compressed_size


class SignSGDClient(BaseCompressionClient):
    def __init__(self, get_model: Callable, server: SignSGDServer, options: FedAvgClientOptions):
        all_options = BaseCompressionClientOptions(
            **options.__dict__,
            compressor=SignCompressor()
        )
        super(SignSGDClient, self).__init__(get_model, server, all_options, residual_accumulation=False)


if __name__ == '__main__':
    x = SignCompressor().compress(torch.tensor([0., 1., -1., 3., 0., 0., 0., 10000.]).cuda())
    print(x)