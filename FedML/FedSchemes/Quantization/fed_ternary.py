import numpy as np
import torch
from torch import nn
from typing import Union, Callable, List, Tuple, Any
from dataclasses import dataclass

from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgServerOptions
from FedML.FedSchemes.Quantization.ternary_weight_network import BaseQuantization
from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function, count_parameters
from FedML.Models.SpecialModels.trainable_ternarize_models import TrainableTernarizedModel


@dataclass
class FedTernServerOptions(FedAvgServerOptions):
    ternarize_client: Callable[[Any], Tuple[List[torch.Tensor], float]]
    ternarize_server: Callable[[Any], Tuple[List[torch.Tensor], float]]


class TrainableTernary(BaseQuantization):
    """
    From paper 'Ternary Compression for Communication-Efficient Federated Learning'
    """
    def quantize(self, x: torch.Tensor):
        delta = 0.05 * torch.max(torch.abs(x))
        alpha_pos = torch.sum(x[torch.greater(x, delta)]) / torch.sum(torch.greater(x, delta))
        alpha_neg = torch.sum(x[torch.greater(-x, delta)]) / torch.sum(torch.greater(-x, delta))
        ternary_x = torch.zeros_like(x)
        ternary_x[x > delta] = alpha_pos
        ternary_x[x < -delta] = alpha_neg
        return ternary_x

    def global_ternarize(self, tensors: Union[List[torch.Tensor]], excluded_indices: List[int] = None):
        return self.quantize_tensor_list(tensors, excluded_indices)

    def local_ternarize(self, m: TrainableTernarizedModel):
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
