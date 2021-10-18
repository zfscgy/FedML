import numpy as np
import torch
from torch import nn

from typing import Callable, List
from dataclasses import dataclass

from FedML.Base.Utils import copy_paras, get_tensors_by_function, get_tensors, set_tensors, set_mean_paras
from FedML.FedSchemes.Quantization.ternary_weight_network import BaseTernary
from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgServerOptions, FedAvgClient, FedAvgClientOptions
from FedML.FedSchemes.Quantization.ternary_weight_network import TernaryServerOptions



class STC(BaseTernary):
    """
    From Paper 'Robust and Communication-Efficient Federated Learning from Non-IID Data' (aka STC, Sparse Ternary Compression)
    The sparsed ternary updates are uploaded to the server (using Golomb encoding to encode non-zero entries)
    The server broadcast the global model update to ALL clients.
    """
    def __init__(self, sparse_rate: float):
        self.sparse_rate = sparse_rate

    def ternarize(self, x: torch.Tensor):
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

        M = round(-1 / np.log2(1 - self.sparse_rate))
        b = round(np.log2(M))
        estimated_compression_ratio = self.sparse_rate * (b + 1 / (1 - np.power(1 - self.sparse_rate, 2**b)))
        compressed_size = len * estimated_compression_ratio / 32 + 1  # The unit is float, not bit

        return ternarized_x, compressed_size


@dataclass
class STCClientOptions(FedAvgClientOptions):
    ternarize: Callable[[nn.Module], List[torch.Tensor]]


class STCClient(FedAvgClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: FedAvgServer, options: STCClientOptions):
        super(STCClient, self).__init__(get_model, server, options)
        self.current_update = 0
        self.unsent_updates = get_tensors_by_function([self.local_model], lambda xs: xs[0] * 0)

    def update(self):
        # Save previous model
        previous_model = get_tensors(self.local_model, copy=True)
        losses = super(STCClient, self).update()
        self.current_update = get_tensors_by_function([self.local_model, previous_model], lambda xy: xy[0] - xy[1])

        # Restore previous model
        set_tensors(self.local_model, previous_model)
        return losses

    def set_ternarized_update(self, update: List[torch.Tensor]):
        new_model = get_tensors_by_function([self.local_model, update], lambda xy: xy[0] + xy[1])
        set_tensors(self.local_model, new_model)

    def get_ternarized_update(self):
        ternarized_update, ternarized_size = \
            self.options.ternarize(get_tensors_by_function([self.current_update, self.unsent_updates],
                                                           lambda xy: xy[0] + xy[1]))

        self.unsent_updates = get_tensors_by_function([self.unsent_updates, self.current_update, ternarized_update],
                                                      lambda xyz: xyz[0] + xyz[1] - xyz[2])
        # print(f"Unsent updates[0]: {torch.std(self.unsent_updates[0]).item():.6f}")
        return ternarized_update, ternarized_size


@dataclass
class STCServerOptions(FedAvgServerOptions):
    ternarize: Callable[[nn.Module], List[torch.Tensor]]


class STCServer(FedAvgServer):
    def __init__(self, get_model: Callable[[], nn.Module], options: STCServerOptions):
        super(STCServer, self).__init__(get_model, options)
        self.current_global_epoch = 0
        self.current_global_batch = 0
        self.current_waiting_list = None
        self.sent_values = None

    def start(self):
        self.current_waiting_list = np.random.permutation(len(self.clients))
        for client in self.clients:
            self.sended_size += copy_paras(self.global_model, client.local_model)
        self.sent_values = get_tensors(self.global_model, copy=True)


    def update(self):
        n_batches_per_epoch = int(np.ceil(len(self.clients) / self.options.n_clients_per_round))
        chosen_clients = self.current_waiting_list[:self.options.n_clients_per_round]  # Randomly choose some clients
        chosen_clients = [self.clients[c] for c in chosen_clients]
        client_updates = []  # Store client updates
        self.current_waiting_list = self.current_waiting_list[self.options.n_clients_per_round:]

        for client in chosen_clients:
            client.update()

            ternarized_client_update, update_size = client.get_ternarized_update()
            client_updates.append(ternarized_client_update)
            self.received_size += update_size

        update_mean = get_tensors_by_function(client_updates, lambda xs: torch.mean(torch.stack(xs, dim=0), dim=0))
        new_global_model = get_tensors_by_function([update_mean, self.global_model], lambda xy: xy[0] + xy[1])
        set_tensors(self.global_model, new_global_model)

        update_full = get_tensors_by_function([new_global_model, self.sent_values], lambda xy: xy[0] - xy[1])
        # print(f"Update_full[0] std: {torch.std(update_full[0]).item():6f}")

        update_send_to_client, update_size = self.options.ternarize(update_full)
        for client in self.clients:
            # diff = get_tensors_by_function([self.sent_values, client.local_model], lambda xy: torch.std(xy[0] - xy[1]))
            # print(f"Diff[0] std: {diff[0].item():6f}")

            client.set_ternarized_update(update_send_to_client)

            self.sended_size += update_size

        self.sent_values = get_tensors_by_function([update_send_to_client, self.sent_values],
                                                   lambda xy: xy[0] + xy[1])

        self.current_global_batch += 1
        if self.current_global_batch == n_batches_per_epoch:
            self.current_global_epoch += 1
            self.current_global_batch = 0
            self.current_waiting_list = np.random.permutation(len(self.clients))  # Re-shuffle the training orders


if __name__ == '__main__':
    a = torch.normal(0, 1, [1000])
    ternarized_a, compressed_size = STC(0.01).ternarize(a)
    print(compressed_size)
    print(ternarized_a)