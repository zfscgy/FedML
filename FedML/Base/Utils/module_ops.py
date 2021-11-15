import torch
from torch import nn
from typing import List, Union, Iterable, Callable


def count_parameters(paras: Union[Iterable[nn.Parameter], nn.Module]):
    if isinstance(paras, nn.Module):
        paras = paras.parameters()
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
            p.data = t
    return count_parameters(paras)


def compute_tensors(func: Callable, *args):
    return [func(*ts) for ts in zip(*args)]


def compute_lists_of_tensors(input_tensors: List[Union[nn.Module, Iterable[nn.Parameter], List[torch.Tensor]]],
                             func: Callable[[List[torch.Tensor]], torch.Tensor]):
    for i, p in enumerate(input_tensors):
        if not isinstance(input_tensors[i], List):
            input_tensors[i] = get_tensors(input_tensors[i])

    result_tensors = []
    for tensors in zip(*input_tensors):
        result_tensors.append(func(list(tensors)))
    return result_tensors


def mean_tensors(input_tensors):
    return compute_lists_of_tensors(input_tensors, lambda xs: torch.mean(torch.stack(xs, dim=0), dim=0))


if __name__ == '__main__':
    x = mean_tensors([[torch.tensor(1.), torch.tensor(2.)], [torch.tensor(3.),  torch.tensor(4.)]])
    print(x)