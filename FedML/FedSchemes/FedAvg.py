from typing import Callable
from dataclasses import dataclass
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from FedML.Base.base import Client, Server
from FedML.Base.Utils import Convert, set_mean_paras, copy_paras, train_n_batches, train_n_epochs


@dataclass
class FedAvgServerOptions:
    n_clients_per_round: int


class FedAvgServer(Server):
    def __init__(self, get_model: Callable[[], nn.Module], options: FedAvgServerOptions):
        super(FedAvgServer, self).__init__(get_model, options)
        self.current_training_clients = []
        self.sended_size = 0
        self.received_size = 0

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
        clients = np.random.choice(self.clients, self.options.n_clients_per_round)
        self.current_training_clients = clients
        for client in clients:
            self.sended_size += copy_paras(self.global_model, client.local_model)
            client.update()

        self.received_size += set_mean_paras([client.local_model for client in clients], self.global_model)


@dataclass
class FedAvgClientOptions:
    client_data_loader: DataLoader
    get_optimizer: Callable[[nn.Module], Optimizer]
    loss_func: Callable
    batch_mode: bool  # If True, use the n_local_batches param, else use the n_local_epochs params
    n_local_rounds: int


class FedAvgClient(Client):
    def __init__(self, get_model: Callable[[], nn.Module], server: FedAvgServer, options: FedAvgClientOptions):
        super(FedAvgClient, self).__init__(get_model, server, options)
        self.optimizer = options.get_optimizer(self.local_model)

        self.train_data_iterator = None

    def update(self):
        if self.options.batch_mode:
            if self.train_data_iterator is None:
                self.train_data_iterator = iter(self.options.client_data_loader)
            self.train_data_iterator, losses = \
                train_n_batches(self.local_model, self.optimizer, self.options.loss_func,
                                self.options.client_data_loader,
                                self.train_data_iterator, self.options.n_local_rounds)
        else:
            losses = train_n_epochs(self.local_model, self.optimizer, self.options.loss_func,
                                    self.options.client_data_loader, self.options.n_local_rounds)
        return losses
