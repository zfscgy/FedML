import numpy as np
import torch
from torch import nn
from typing import Union, Callable, List, Tuple, Any
from dataclasses import dataclass


from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgServerOptions
from FedML.FedSchemes.Quantization.base_quantization import BaseQuantization
from FedML.Base.Utils import get_tensors, set_tensors, get_tensors_by_function, count_parameters



class NaiveTernary(BaseQuantization):
    """
    From paper 'Ternary Weight Networks'
    """
    def quantize(self, x: torch.Tensor):
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
