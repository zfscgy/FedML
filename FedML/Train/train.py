import logging
from typing import List, Callable
from torch.utils.data import DataLoader

from FedML.Base.Utils import test_on_data_loader
from FedML.Base.base import Server
from FedML.Base.config import GlobalConfig

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
        test_records = []
        for i in range(n_global_rounds):
            if i % test_per_global_rounds == 0:
                test_metric_values = test_on_data_loader(self.server.global_model, test_data_loader, test_metrics,
                                                         GlobalConfig.fast_mode)
                test_records.append([i] + test_metric_values)
                format_res = "Metrics: " + " ".join(f"{metric:.3f}" for metric in test_metric_values)
                logger.info(f"round {i} " + format_res)
                if test_callback is not None:
                    test_callback()
            logger.info(f"Round {i} begins===========")
            self.update()
            if round_callback is not None:
                round_callback()
            logger.info(f"Round {i} ends=========")
        return test_records


__all__ = ["FedTrain"]