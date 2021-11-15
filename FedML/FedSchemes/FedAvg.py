from dataclasses import dataclass
from typing import Callable

import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from FedML.Base.Utils import get_tensors, mean_tensors, set_tensors, train_n_batches, train_n_epochs
from FedML.Base.base import Client, Server
from FedML.Base.config import GlobalConfig


@dataclass
class FedAvgServerOptions:
    n_clients_per_round: int


class FedAvgServer(Server):
    def __init__(self, get_model: Callable[[], nn.Module], options: FedAvgServerOptions):
        super(FedAvgServer, self).__init__(get_model, options)

    def start(self):
        """
        At the beginning of FedAVG training, all clients' model are identically initialized
        :return:
        """
        pass

    def update(self):
        """
        Choose n_clients_per_round clients to train their local models,
        then get the average of those local models
        :return:
        """
        clients = np.random.choice(self.clients, self.options.n_clients_per_round, replace=False)
        self.current_training_clients = clients
        client_new_models = []
        for client in clients:
            self.sended_size += set_tensors(self.global_model, client.local_model, copy=True)
            client.update()
            client_new_models.append(get_tensors(client.local_model, copy=True))

        self.received_size += set_tensors(self.global_model, mean_tensors(client_new_models))
        self.current_global_rounds += 1


@dataclass
class FedAvgClientOptions:
    client_data_loader: DataLoader
    get_optimizer: Callable[[nn.Module], Optimizer]
    loss_func: Callable
    batch_mode: bool  # If True, use the n_local_batches param, else use the n_local_epochs params
    n_local_rounds: int


class FedAvgClient(Client):
    def __init__(self, get_model: Callable[[], nn.Module], server: Server, options: FedAvgClientOptions):
        super(FedAvgClient, self).__init__(get_model, server, options)

        self.train_data_iterator = None
        self.optimizer = None

    def update(self):
        optimizer = self.options.get_optimizer(self.local_model)
        if self.options.batch_mode:
            if self.train_data_iterator is None:
                self.train_data_iterator = iter(self.options.client_data_loader)
            losses, self.train_data_iterator = \
                train_n_batches(self.local_model, optimizer, self.options.loss_func,
                                self.options.client_data_loader,
                                self.train_data_iterator, self.options.n_local_rounds,
                                GlobalConfig.fast_mode)
        else:
            losses = train_n_epochs(self.local_model, optimizer, self.options.loss_func,
                                    self.options.client_data_loader, self.options.n_local_rounds,
                                    GlobalConfig.fast_mode)
        return losses
