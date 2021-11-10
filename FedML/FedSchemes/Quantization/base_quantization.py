import torch
import numpy as np

from typing import List



class BaseQuantization:
    def quantize(self, x: torch.Tensor) -> [torch.Tensor, float]:
        """
        :param x:
        :return: Quantized tensor, compression rate
        """
        raise NotImplementedError()


def quantize_by_batch(quantizer, x: torch.Tensor, n_batch_dims=0):
    flat_to_shape = x.shape[n_batch_dims:]
    reshaped_x = x.reshape(-1, *flat_to_shape)
    ternarized_x = torch.zeros_like(reshaped_x)
    ternarized_size = 0
    for i in range(reshaped_x.shape[0]):
        ternarized_x[i], compressed_size = quantizer.quantize(reshaped_x[i])
        ternarized_size += ternarized_size
    return ternarized_x.view(*x.shape), ternarized_size

def quantize_tensor_list(quantizer, tensors: List[torch.Tensor], excluded_indices: List[int] = None):
    if excluded_indices is None:
        excluded_indices = []
    compressed_size = 0
    ternaried_tensors = []
    for i in range(len(tensors)):
        if i not in excluded_indices:
            ternarized_batch, ternarized_size = quantizer.quantize_by_batch(tensors[i], n_batch_dims=1)
            ternaried_tensors.append(ternarized_batch)
            compressed_size += ternarized_size
            # log2(3) since only 3 values are needed
        else:
            ternaried_tensors.append(tensors[i])
            compressed_size += np.prod(list(tensors[i].size()))
    return ternaried_tensors, compressed_size