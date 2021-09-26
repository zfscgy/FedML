import numpy as np
import torch
from torch import nn
from typing import Union, Callable, List, Tuple
from dataclasses import dataclass

from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgClient, FedAvgServerOptions, FedAvgClientOptions
from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function, count_parameters


def ternarize(x: torch.Tensor):
    delta = 0.7 * torch.mean(torch.abs(x))
    alpha = torch.sum(torch.abs(x)[torch.greater(torch.abs(x), delta)]) / torch.sum(torch.greater(torch.abs(x), delta))
    ternary_x = torch.zeros_like(x)
    ternary_x[x > delta] = alpha
    ternary_x[x < -delta] = -alpha
    return ternary_x


def ternarize_by_batch(x: torch.Tensor, n_batch_dims=0):
    """
    In the paper "Ternary network", the tensors are not entirely ternarized, but splitted into pieces,
    and being ternarized within each piece.
    e.g. [[-1, 0, 1], [2, 0, -3]] are ternarized into [[-1, 0, 1], [2.5, 0, 2.5]]
    :param x:
    :param n_batch_dims:
    :return:
    """
    flat_to_shape = x.shape[n_batch_dims:]
    reshaped_x = x.reshape(-1, *flat_to_shape)
    ternarized_x = torch.zeros_like(reshaped_x)
    for i in range(reshaped_x.shape[0]):
        ternarized_x[i] = ternarize(reshaped_x[i])
    return ternarized_x.view(*x.shape)


def ternarize_tensor_list(tensors: List[torch.Tensor], excluded_indices: List[int] = None):
    compressed_size = 0
    ternaried_tensors = []
    for i in range(len(tensors)):
        if i not in excluded_indices:
            ternaried_tensors.append(ternarize_by_batch(tensors[i], n_batch_dims=1))
            compressed_size += np.prod(list(tensors[i].size())) * 1.585 / 32 + 1
            # log2(3) since only 3 values are needed
        else:
            ternaried_tensors.append(tensors[i])
            compressed_size += np.prod(list(tensors[i].size()))
    return ternaried_tensors, compressed_size / sum(np.prod(list(update.size()))for update in tensors)


def ternarize_except_last_linear(tensors: List[torch.Tensor]):
    n_tensors = len(tensors)
    return ternarize_tensor_list(tensors, [n_tensors - 2, n_tensors - 1])


@dataclass
class TernaryServerOptions(FedAvgServerOptions):
    ternarize: Callable[[List[torch.Tensor]], List[torch.Tensor]]


class TernaryServer(FedAvgServer):
    """
    Naive Ternary Weight Network
    """
    def __init__(self, get_model: Callable[[], nn.Module], options: TernaryServerOptions):
        super(TernaryServer, self).__init__(get_model, options)
        self.compress_rate = self.options.ternarize(get_tensors(self.global_model))[1]

    def update(self):
        clients = np.random.choice(self.clients, self.options.n_clients_per_round)
        self.current_training_clients = clients
        ternarized_global_model = self.options.ternarize(get_tensors(self.global_model))[0]
        # ternarized_global_model = get_tensors(self.global_model, copy=True)
        received_updates = []
        for client in clients:
            self.sended_size += set_tensors(client.local_model, ternarized_global_model, copy=True) * \
                                self.compress_rate
            client.update()
            trained_weight = get_tensors(client.local_model, copy=True)
            ternarized_update = self.options.ternarize([w1 - w0 for w1, w0 in zip(trained_weight, ternarized_global_model)])[0]
            received_updates.append(ternarized_update)

        initial_global_weights = get_tensors(self.global_model)
        mean_update = get_tensors_by_function(received_updates, lambda xs: torch.mean(torch.stack(xs, dim=0), dim=0))
        set_tensors(self.global_model, get_tensors_by_function([initial_global_weights, mean_update], lambda xs: xs[0] + xs[1]))

        self.received_size += len(clients) * count_parameters(self.global_model.parameters()) * self.compress_rate


class TrainableTernaryServer(FedAvgServer):
    """
    Using the method described in the paper 'Ternary Compression For Communication-Efficient Federated Learning'
    """
    def __init__(self, get_model: Callable[[], nn.Module], options: TernaryServerOptions):
        super(TrainableTernaryServer, self).__init__(get_model, options)
        self.compress_rate = self.options.ternarize(get_tensors(self.global_model))[1]

    def update(self):
        clients = np.random.choice(self.clients, self.options.n_clients_per_round)
        self.current_training_clients = clients
        ternarized_global_model = self.options.ternarize(get_tensors(self.global_model))[0]
        received_weights = []
        for client in clients:
            self.sended_size += set_tensors(client.local_model, ternarized_global_model, copy=True) * \
                                self.compress_rate
            client.update()
            trained_weights = client.local_model.get_ternarized_parameters()
            received_updates.append(ternarized_update)

        initial_global_weights = get_tensors(self.global_model)
        mean_update = get_tensors_by_function(received_updates, lambda xs: torch.mean(torch.stack(xs, dim=0), dim=0))
        set_tensors(self.global_model, get_tensors_by_function([initial_global_weights, mean_update], lambda xs: xs[0] + xs[1]))

        self.received_size += len(clients) * count_parameters(self.global_model.parameters()) * self.compress_rate


class TrainableTernaryClient(FedAvgClient):
    pass





if __name__ == '__main__':
    x = ternarize_by_batch(torch.normal(0, 1, [100, 10]), n_batch_dims=1)
    print(x.numpy())
