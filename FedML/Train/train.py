import logging
from typing import List, Callable
import numpy as np
from torch.utils.data import DataLoader

from FedML.Base.Utils import Convert, test_on_data_loader
from FedML.Base.base import Server, Client


logging.basicConfig(format="%(asctime)-15s %(message)s")
logger = logging.getLogger("Train")
logger.setLevel(logging.INFO)


class FedTrain:
    def __init__(self, server: Server):
        self.server = server
        self.clients = server.clients

    def start(self):
        self.server.start()

    def update(self):
        self.server.update()

    def train(self, n_global_rounds: int, test_per_global_rounds: int,
              test_data_loader: DataLoader, test_metrics: List[Callable],
              round_callback: Callable = None, test_callback: Callable = None):
        for i in range(n_global_rounds):
            if i % test_per_global_rounds == 0:
                test_metric_values = test_on_data_loader(self.server.global_model, test_data_loader, test_metrics)
                format_res = "Metrics: " + " ".join(f"{metric:.3f}" for metric in test_metric_values)
                logger.info(f"round {i} " + format_res)
                if test_callback is not None:
                    test_callback()
            logger.info(f"Round {i} begins===========")
            self.update()
            if round_callback is not None:
                round_callback()
            logger.info(f"Round {i} ends=========")


__all__ = ["FedTrain"]
