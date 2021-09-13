import torch
from torch import nn

from typing import List, List


import numpy as np
import torch

from typing import Union


class Convert:
    cuda = True

    @staticmethod
    def to_numpy(x: torch.Tensor):
        try:
            return x.cpu().detach().numpy()
        except:
            pass

        return x.cpu().numpy()

    @staticmethod
    def to_tensor(x: Union[torch.Tensor, np.ndarray]):
        if isinstance(x, torch.Tensor):
            if Convert.cuda:
                return x.cuda()
            else:
                return x
        else:
            if Convert.cuda:
                return torch.tensor(x).cuda()
            else:
                return x


def count_parameters(paras: List[nn.Parameter]):
    return sum(p.numel() for p in paras)


# Copy parameters from source to target
def copy_paras(source_params: List[nn.Parameter], target_params: List[nn.Parameter], add: bool = False):
    """
    :param source_params:
    :param target_params:
    :param add: Add the source params to target param
    :return: Number of elements copied
    """
    for source_w, target_w in zip(source_params, target_params):
        if add:
            target_w.data += torch.clone(source_w.data)
        else:
            target_w.data = torch.clone(source_w.data)
    return count_parameters(source_params)


def copy_model_paras(source_model: nn.Module, target_model: nn.Module, add: bool = False):
    return copy_paras(list(source_model.parameters()), list(target_model.parameters()), add)


# Copy parameters mean from multiple sources to target
def copy_mean_paras(source_paras: List[List[nn.Parameter]], target_para: List[nn.Parameter], add: bool = False):
    for i, p in enumerate(target_para):
        if add:
            p.data += torch.mean(torch.cat([source_para[i].data for source_para in source_paras], dim=0), dim=0)
        else:
            p.data = torch.mean(torch.cat([source_para[i].data for source_para in source_paras], dim=0), dim=0)
    return count_parameters(sum(source_paras, start=[]))


def copy_mean_model(source_models: List[nn.Module], target_model, add: bool = False):
    return copy_mean_paras([list(m.parameters()) for m in source_models], target_model, add)
