import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List, Iterator, Callable

from FedML.Base.Utils.convert import Convert



def train_n_batches(model: nn.Module, optimizer: Optimizer, loss_func: Callable,
                    data_loader: DataLoader, data_iterator: Iterator, n_batches: int):

    cpu_model = model
    model = Convert.model_to_device(cpu_model)

    i = 0
    losses = []
    while i < n_batches:
        try:
            xs, ys = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            xs, ys = next(data_iterator)
        pred_ys = model(Convert.to_tensor(xs))
        loss = loss_func(pred_ys, Convert.to_tensor(ys))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        i += 1

    Convert.model_to_cpu(model, cpu_model)

    return losses, data_iterator


def train_n_epochs(model: nn.Module, optimizer: Optimizer, loss_func: Callable,
                   data_loader: DataLoader, n_epochs: int):
    cpu_model = model
    model = Convert.model_to_device(cpu_model)

    losses = []
    for i in range(n_epochs):
        for xs, ys in data_loader:
            pred_ys = model(Convert.to_tensor(xs))
            loss = loss_func(pred_ys, Convert.to_tensor(ys))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    Convert.model_to_cpu(model, cpu_model)
    return losses


def test_on_data_loader(model: nn.Module, data_loader: DataLoader, metrics: List[Callable]):
    model = Convert.model_to_device(model)

    ys = []
    pred_ys = []
    for batch_xs, batch_ys in data_loader:
        batch_pred_ys = model(Convert.to_tensor(batch_xs))
        ys.append(Convert.to_numpy(batch_ys))
        pred_ys.append(Convert.to_numpy(batch_pred_ys))

    ys = np.concatenate(ys, axis=0)
    pred_ys = np.concatenate(pred_ys, axis=0)

    metric_values = [metric(pred_ys, ys) for metric in metrics]
    model.to('cpu')
    return metric_values
