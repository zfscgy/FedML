import numpy as np
import torch
from torch import nn
from typing import Union, Callable, List, Tuple, Any
from dataclasses import dataclass

from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgServerOptions
from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function, count_parameters


class BaseTernary:
    def ternarize(self, x: torch.Tensor) -> [torch.Tensor, float]:
        raise NotImplementedError()

    def ternarize_by_batch(self, x: torch.Tensor, n_batch_dims=0):
        flat_to_shape = x.shape[n_batch_dims:]
        reshaped_x = x.reshape(-1, *flat_to_shape)
        ternarized_x = torch.zeros_like(reshaped_x)
        ternarized_size = 0
        for i in range(reshaped_x.shape[0]):
            ternarized_x[i], compressed_size = self.ternarize(reshaped_x[i])
            ternarized_size += ternarized_size
        return ternarized_x.view(*x.shape), ternarized_size

    def ternarize_tensor_list(self, tensors: List[torch.Tensor], excluded_indices: List[int] = None):
        if excluded_indices is None:
            excluded_indices = []
        compressed_size = 0
        ternaried_tensors = []
        for i in range(len(tensors)):
            if i not in excluded_indices:
                ternarized_batch, ternarized_size = self.ternarize_by_batch(tensors[i], n_batch_dims=1)
                ternaried_tensors.append(ternarized_batch)
                compressed_size += ternarized_size
                # log2(3) since only 3 values are needed
            else:
                ternaried_tensors.append(tensors[i])
                compressed_size += np.prod(list(tensors[i].size()))
        return ternaried_tensors, compressed_size


class NaiveTernary(BaseTernary):
    """
    From paper 'Ternary Weight Networks'
    """
    def ternarize(self, x: torch.Tensor):
        delta = 0.7 * torch.mean(torch.abs(x))
        alpha = torch.sum(torch.abs(x)[torch.greater(torch.abs(x), delta)]) / torch.sum(torch.greater(torch.abs(x), delta))
        ternary_x = torch.zeros_like(x)
        ternary_x[x > delta] = alpha
        ternary_x[x < -delta] = -alpha
        return ternary_x, np.prod(list(x.size())) * np.log2(3) / 32+ 1


@dataclass
class TernaryServerOptions(FedAvgServerOptions):
    ternarize_client: Callable[[Any], Tuple[List[torch.Tensor], float]]
    ternarize_server: Callable[[Any], Tuple[List[torch.Tensor], float]]


class TernaryServer_GradientAvg(FedAvgServer):
    """
    Naive Ternary Weight Network
    """
    def __init__(self, get_model: Callable[[], nn.Module], options: TernaryServerOptions):
        super(TernaryServer_GradientAvg, self).__init__(get_model, options)
        self.compress_rate = self.options.ternarize_server(get_tensors(self.global_model))[1]

    def update(self):
        clients = np.random.choice(self.clients, self.options.n_clients_per_round)
        self.current_training_clients = clients
        ternarized_global_model = self.options.ternarize_server(get_tensors(self.global_model))[0]
        # ternarized_global_model = get_tensors(self.global_model, copy=True)
        received_updates = []
        for client in clients:
            self.sended_size += set_tensors(client.local_model, ternarized_global_model, copy=True) * \
                                self.compress_rate
            client.update()
            trained_weight = get_tensors(client.local_model, copy=True)
            ternarized_update = self.options.ternarize_client([w1 - w0 for w1, w0 in zip(trained_weight, ternarized_global_model)])[0]
            received_updates.append(ternarized_update)

        initial_global_weights = get_tensors(self.global_model)
        mean_update = get_tensors_by_function(received_updates, lambda xs: torch.mean(torch.stack(xs, dim=0), dim=0))
        set_tensors(self.global_model, get_tensors_by_function([initial_global_weights, mean_update], lambda xs: xs[0] + xs[1]))

        self.received_size += len(clients) * count_parameters(self.global_model.parameters()) * self.compress_rate
