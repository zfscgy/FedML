import logging
from typing import List, Callable
from dataclasses import dataclass

import numpy as np

from torch.utils.data import DataLoader


from FedML.Base.utils import Convert
from FedML.Base.base import Server, Client


logging.basicConfig()
logger = logging.getLogger("Train")
logger.setLevel(logging.INFO)


@dataclass
class TrainOptions:
    n_global_rounds: int
    test_per_global_epochs: int
    test_dataloader: DataLoader
    test_metrics: List[Callable]


def fed_train(server: Server, clients: List[Client], options: TrainOptions):
    for i in range(options.n_global_rounds):
        if i % options.test_per_global_epochs:
            ys = []
            pred_ys = []
            for batch_xs, batch_ys in options.test_dataloader:
                batch_pred_ys = server.global_model(batch_xs)
                ys.append(Convert.to_numpy(batch_ys))
                pred_ys.append(Convert.to_numpy(batch_pred_ys))

            ys = np.concatenate(ys, axis=0)
            pred_ys = np.concatenate(pred_ys, axis=0)

            format_res = "Metrics: " + " ".join(f"{metric(pred_ys, ys):.3f}" for metric in options.test_metrics)
            logger.info(format_res)

        for client in clients:
            client.update()
        server.update()
