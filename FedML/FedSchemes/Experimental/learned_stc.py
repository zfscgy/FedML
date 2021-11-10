import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Union, Callable, List
from dataclasses import dataclass

from FedML.Base.Utils import get_tensors_by_function, get_tensors, set_tensors, \
    count_parameters, test_on_data_loader, Convert
from FedML.Base.config import GlobalConfig
from FedML.FedSchemes.fedavg import FedAvgServer, FedAvgClient
from FedML.FedSchemes.Quantization.sparse_ternary_compression import STCClient, STCClientOptions, \
    STCServerOptions, STCServer



def l1_dist(model: nn.Module, ws: List[torch.Tensor]):
    return sum([torch.sum(torch.abs(p - w)) for p, w in zip(model.parameters(), ws)])


class SmoothL1(nn.Module):
    def __init__(self, threshold: Union[torch.Tensor, float]):
        super(SmoothL1, self).__init__()
        self.threshold = threshold
        self.t_square = torch.square(self.threshold)

    def forward(self, x):
        abs_xs = torch.abs(x)
        big_xs = (abs_xs >= self.threshold)
        distances = torch.where(big_xs, abs_xs - 0.5 * self.threshold, 0.5 * torch.square(x) / self.threshold)
        return torch.sum(distances)


@dataclass
class LearnedSTCServerOptions(STCServerOptions):
    residual_shrinkage: float
    dynamic_shrinkage: bool
    ref_data_loader: DataLoader
    ref_metric: Callable


class LearnedSTCServer(STCServer):
    def __init__(self, get_model: Callable[[], nn.Module], options: LearnedSTCServerOptions):
        super(LearnedSTCServer, self).__init__(get_model, options)
        self.last_val_metric = None

    def update(self):
        self.unsent_values = get_tensors_by_function([self.unsent_values], lambda x: (1 - self.options.residual_shrinkage) * x[0])

        super(LearnedSTCServer, self).update()
        if self.options.dynamic_shrinkage:
            val_metric = test_on_data_loader(self.global_model, self.options.ref_data_loader,
                                             [self.options.ref_metric],
                                             GlobalConfig.fast_mode)[0]
            if self.last_val_metric is not None:
                if val_metric <= self.last_val_metric:
                    self.options.residual_shrinkage *= 0.99
                else:
                    self.options.residual_shrinkage *= 1 / 0.99
                self.options.residual_shrinkage = np.clip(self.options.residual_shrinkage, 0.5, 0.99)
                print(f"Current val metric {val_metric:.4f}, residual shrinkage: {self.options.residual_shrinkage:.4f}")

            self.last_val_metric = val_metric


@dataclass
class LearnedSTCClientOptions(STCClientOptions):
    residual_shrinkage: float
    lambda_l1: float
    smooth_l1: float
    dynamic_l1: bool




class LearnedSTCClient(STCClient):
    def __init__(self, get_model: Callable[[], nn.Module], server: FedAvgServer, options: LearnedSTCClientOptions):
        super(LearnedSTCClient, self).__init__(get_model, server, options)
        self.options.base_loss = self.options.loss_func
        self.options.base_lambda_l1 = self.options.lambda_l1

        self.loss_diff_records = []


    def update(self):
        # Save previous model
        self.unsent_updates = get_tensors_by_function(
            [self.unsent_updates], lambda xs: xs[0] * self.options.residual_shrinkage)

        previous_model = get_tensors(self.local_model, copy=True)

        previous_model_sub_unsent = get_tensors_by_function(
            [previous_model, self.unsent_updates], lambda xs: xs[0] - xs[1])
        if not self.options.smooth_l1:
            self.options.loss_func = lambda pred_ys, ys: \
                self.options.base_loss(pred_ys, ys) + \
                self.options.lambda_l1 * l1_dist(self.local_model, [Convert.to_tensor(t) for t in previous_model_sub_unsent])
        else:
            # Calculate smooth thresholds
            xs, ys = next(iter(self.options.client_data_loader))
            if GlobalConfig.fast_mode:
                xs = Convert.to_tensor(xs)
                ys = Convert.to_tensor(ys)
            loss = self.options.base_loss(self.local_model(xs), ys)
            loss.backward()
            grad_thresholds = []
            for p in self.local_model.parameters():
                k = int(0.01 * int(p.grad.flatten().shape[0])) or 1
                top_p_xs, _ = torch.topk(torch.abs(p.grad.flatten()), k, sorted=True)
                grad_thresholds.append(top_p_xs[-1])
            self.local_model.zero_grad()

            smooth_l1s = []
            for p, t in zip(self.local_model.parameters(), grad_thresholds):
                smooth_l1s.append(SmoothL1(0.01 * t))

            def loss_func(x, y):
                loss = self.options.base_loss(x, y)
                for s, p, pi in zip(smooth_l1s, self.local_model.parameters(), previous_model):
                    loss += self.options.lambda_l1 * s(p - pi)
                return loss
            self.options.loss_func = loss_func

        losses = FedAvgClient.update(self)

        loss_diff = np.mean(losses[:10]) - np.mean(losses[-10:])
        self.loss_diff_records.append(loss_diff)

        print(f"Train losses (first/last 10): {np.mean(losses[:10]):.6f} {np.mean(losses[-10:]):.6f}")
        self.current_update = get_tensors_by_function([self.local_model, previous_model], lambda xy: xy[0] - xy[1])
        # Restore previous model
        set_tensors(self.local_model, previous_model)


        if self.options.dynamic_l1:
            if len(self.loss_diff_records) > 1:
                lambda_l1 = self.options.base_lambda_l1 * \
                            np.max([self.loss_diff_records[-1], 0.001]) / np.mean(np.maximum(self.loss_diff_records, 0.001))
                print(f"lambda_l1 adjusted: {lambda_l1:.6f}")
                self.options.lambda_l1 = lambda_l1

        return losses


    def get_ternarized_update(self):
        # print(f"Client unsent: {torch.std(self.unsent_updates[0]).item():.6f}")
        return super(LearnedSTCClient, self).get_ternarized_update()


