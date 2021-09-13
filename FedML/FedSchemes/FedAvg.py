from typing import Callable
from dataclasses import dataclass
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from FedML.Base.base import Client, Server
from FedML.Base.utils import copy_model_paras


@dataclass
class FedAvgServerOptions:
    n_clients_per_round: int


class FedAvgServer(Server):
    def __init__(self, get_model: Callable[[], nn.Module], options):
        super(FedAvgServer, self).__init__(get_model, options)
        self.sended_size = 0
        self.received_size = 0

    def start(self):
        """
        At the beginning of FedAVG training, all clients' model are identically initialized
        :return:
        """
        for client in self.clients:
            self.sended_size += copy_model_paras(self.global_model, client.local_model)

    def update(self):
        """
        Choose n_clients_per_round clients to train their local models,
        then get the average of those local models
        :return:
        """
        clients = np.random.choice(self.clients, self.options.n_clients_per_round)
        for client in clients:
            client.update()


@dataclass
class FedAvgClientOptions:
    client_dataloader: DataLoader
    get_optimizer: Callable[[nn.Module], Optimizer]
    loss_func: Callable
    batch_mode: bool  # If True, use the n_local_batches param, else use the n_local_epochs params
    n_local_batches: int
    n_local_epochs: int


class FedAvgClient(Client):
    def __init__(self, get_model: Callable[[], nn.Module], server: FedAvgServer, options: FedAvgClientOptions):
        super(FedAvgClient, self).__init__(get_model, server, options)
        self.optimizer = options.get_optimizer(self.local_model)

    def update(self):
        # Fetch model parameters from server
        copy_model_paras(self.server.global_model, self.local_model)

        losses = []
        if self.options.batch_mode:
            i = 0
            data_loader = iter(self.options.client_dataloader)
            while i < self.options.n_local_batches:
                try:
                    xs, ys = next(data_loader)
                except StopIteration:
                    data_loader = iter(self.options.client_dataloader)
                    xs, ys = next(data_loader)
                pred_ys = self.local_model(ys)
                loss = self.options.loss_func(pred_ys, ys)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return losses
