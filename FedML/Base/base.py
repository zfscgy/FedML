import torch
from torch import nn

from typing import Callable, List


class Server:
    def __init__(self, get_model: Callable[[], nn.Module], options):
        self.global_model = get_model()
        self.clients = []
        self.options = options

    def set_clients(self, clients: List):
        self.clients = clients

    def start(self):
        raise NotImplemented()

    def update(self):
        raise NotImplemented()


class Client:
    def __init__(self, get_model: Callable[[], nn.Module], server: Server, options):
        self.server = server
        self.local_model = get_model()
        self.options = options

    def update(self):
        raise NotImplemented()