import torch
from torch import nn
from typing import Callable, List

from FedML.Base.Utils import set_tensors, get_tensors
from FedML.Base.config import GlobalConfig


class Server:
    """
    Federated learning server
    """
    def __init__(self, get_model: Callable[[], nn.Module], options):
        self.global_model = get_model()
        self.clients = []
        self.options = options

        self.current_training_clients = []
        self.sended_size = 0
        self.received_size = 0
        self.current_global_rounds = 0

    def set_clients(self, clients: List):
        self.clients = clients

    def start(self):
        """
        At the begining, server model is copied to each client
        :return:
        """
        for i, client in enumerate(self.clients):
            if i == 0 or not GlobalConfig.fast_mode:
                set_tensors(client.local_model, get_tensors(self.global_model, copy=True))

    def update(self):
        raise NotImplemented()


class Client:
    def __init__(self, get_model: Callable[[], nn.Module], server: Server, options):
        self.server = server
        self.local_model = get_model()
        self.options = options

    def update(self):
        raise NotImplemented()
