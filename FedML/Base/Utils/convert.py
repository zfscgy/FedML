import numpy as np
import torch
from torch import nn
from typing import Union


from FedML.Base.config import GlobalConfig


class Convert:
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
            return x.to(GlobalConfig.device)
        else:
            return torch.tensor(x).to(GlobalConfig.device)

    @staticmethod
    def model_to_device(m: nn.Module):
        return m.to(GlobalConfig.device)

    @staticmethod
    def model_to_cpu(m_cuda: nn.Module):
        return m_cuda.to('cpu')
