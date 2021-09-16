import torch
from torch import nn
from typing import List, Union, Iterable, Callable


def count_parameters(paras: Iterable[nn.Parameter]):
    return sum(p.numel() for p in paras)


def get_tensors(paras: Union[nn.Module, Iterable[nn.Parameter]], copy: bool = False) -> List[torch.Tensor]:
    if isinstance(paras, nn.Module):
        paras = paras.parameters()

    if not copy:
        tensors = [p.data for p in paras]
    else:
        tensors = [torch.clone(p.data) for p in paras]
    return tensors


def set_tensors(paras: Union[nn.Module, Iterable[nn.Parameter]], tensors: List[torch.Tensor], copy=False):
    if isinstance(paras, nn.Module):
        paras = paras.parameters()
    for p, t in zip(paras, tensors):
        if copy:
            p.data = torch.clone(t)
        else:
            p.data = t.data
    return count_parameters(paras)


def get_tensors_by_function(source_paras: List[Union[nn.Module, Iterable[nn.Parameter], List[torch.Tensor]]],
                            func: Callable[[List[torch.Tensor]], torch.Tensor]):
    if not isinstance(source_paras[0], List) or not isinstance(source_paras[0][0], torch.Tensor):
        source_paras = [get_tensors(source_para) for source_para in source_paras]

    result_tensors = []
    for tensors in zip(*source_paras):
        result_tensors.append(func(list(tensors)))
    return result_tensors


def copy_paras(source_para: Union[Iterable[nn.Parameter], nn.Module],
               target_para: Union[Iterable[nn.Parameter], nn.Module]):
    return set_tensors(target_para, get_tensors(source_para, copy=True))


# Copy parameters mean from multiple sources to target
def set_mean_paras(source_paras: List[Union[Iterable[nn.Parameter], nn.Module]],
                   target_para: Union[Iterable[nn.Parameter], nn.Module]):
    mean_tensor = get_tensors_by_function(source_paras, lambda xs: torch.mean(torch.stack(xs), dim=0))
    return set_tensors(target_para, mean_tensor)
