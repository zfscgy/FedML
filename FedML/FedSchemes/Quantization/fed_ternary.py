import numpy as np
import torch
from torch import nn
from typing import Union, Callable, List, Any
from dataclasses import dataclass

from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgServerOptions
from FedML.FedSchemes.Quantization.ternary_weight_network import ternarize_by_batch
from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function, count_parameters
from FedML.Models.SpecialModels.trainable_ternarize_models import TrainableTernarizedModel


@dataclass
class FedTernServerOptions(FedAvgServerOptions):
    ternarize_client: Callable[[Any], List[torch.Tensor]]
    ternarize_server: Callable[[Any], List[torch.Tensor]]


class TernaryServer_GradientAvg(FedAvgServer):
    """
    Naive Ternary Weight Network
    """
    def __init__(self, get_model: Callable[[], nn.Module], options: FedTernServerOptions):
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


class TrainableTernary:
    """
    From paper 'Ternary Compression for Communication-Efficient Federated Learning'
    """
    @staticmethod
    def ternarize(x: torch.Tensor):
        delta = 0.05 * torch.max(torch.abs(x))
        alpha_pos = torch.sum(x[torch.greater(x, delta)]) / torch.sum(torch.greater(x, delta))
        alpha_neg = torch.sum(x[torch.greater(-x, delta)]) / torch.sum(torch.greater(-x, delta))
        ternary_x = torch.zeros_like(x)
        ternary_x[x > delta] = alpha_pos
        ternary_x[x < -delta] = alpha_neg
        return ternary_x

    @staticmethod
    def ternarize_by_batch(x: torch.Tensor, n_batch_dims=0):
        """
        In the paper "Ternary network", the tensors are not entirely ternarized, but splitted into pieces,
        and being ternarized within each piece.
        e.g. [[-1, 0, 1], [2, 0, -3]] are ternarized into [[-1, 0, 1], [2.5, 0, 2.5]]
        :param x:
        :param n_batch_dims:
        :return:
        """
        return ternarize_by_batch(x, TrainableTernary.ternarize, n_batch_dims)

    @staticmethod
    def global_ternarize(tensors: Union[List[torch.Tensor]], excluded_indices: List[int] = None):
        compressed_size = 0
        ternaried_tensors = []
        for i in range(len(tensors)):
            if i not in excluded_indices:
                ternaried_tensors.append(TrainableTernary.ternarize_by_batch(tensors[i], n_batch_dims=0))
                compressed_size += np.prod(list(tensors[i].size())) * 1.585 / 32 + 2
                # log2(3) since only 3 values are needed, but the w_pos and w_neg are different
            else:
                ternaried_tensors.append(tensors[i])
                compressed_size += np.prod(list(tensors[i].size()))
        return ternaried_tensors, compressed_size / sum(np.prod(list(update.size()))for update in tensors)

    @staticmethod
    def local_ternarize(m: TrainableTernarizedModel):
        return m.get_ternarized_values(), m.get_compression_rate()


class FedTernServer(FedAvgServer):
    """
    Using the method described in the paper 'Ternary Compression For Communication-Efficient Federated Learning'
    """
    def __init__(self, get_model: Callable[[], nn.Module], options: FedTernServerOptions):
        super(FedTernServer, self).__init__(get_model, options)
        self.compress_rate = self.options.ternarize_server(self.global_model.get_ternarized_values())[1]
        self.n_tensor_elements = sum([np.prod(list(p.size())) for p in self.global_model.get_ternarized_values()])

    def update(self):
        clients = np.random.choice(self.clients, self.options.n_clients_per_round)
        self.current_training_clients = clients
        ternarized_global_values = self.options.ternarize_server(self.global_model.get_ternarized_values())[0]
        received_weights = []
        for client in clients:
            self.sended_size += self.n_tensor_elements * self.compress_rate
            client.local_model.set_ternarized_values(ternarized_global_values)
            client.update()
            trained_weights = client.local_model.get_ternarized_values()
            received_weights.append(trained_weights)

        mean_weight = get_tensors_by_function(received_weights, lambda xs: torch.mean(torch.stack(xs, dim=0), dim=0))
        self.global_model.set_ternarized_values(mean_weight)

        self.received_size += len(clients) * count_parameters(self.global_model.parameters()) * self.compress_rate


if __name__ == '__main__':
    pass
