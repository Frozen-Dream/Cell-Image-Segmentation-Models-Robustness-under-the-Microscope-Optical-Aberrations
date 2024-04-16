from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .containers import Parallel


class Residual(nn.Sequential):
    """Pass the input through a layer and add the result with the input."""

    def __init__(self, layer):
        # yapf: disable
        layers = [
            Parallel([layer, nn.Identity()]),
            Sum()
        ]
        # yapf: enable
        super().__init__(*layers)


class ModulizedFunction(nn.Module):
    """Convert a function to an nn.Module."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = partial(fn, *args, **kwargs)

    def forward(self, x):
        return self.fn(x)


class Interpolate(ModulizedFunction):
    def __init__(self, mode='bilinear', align_corners=False, **kwargs):
        super().__init__(
            F.interpolate, mode='bilinear', align_corners=False, **kwargs)


class SplitTensor(nn.Module):
    """ Wrapper around `torch.split` """

    def __init__(self, size_or_sizes, dim):
        super().__init__()
        self.size_or_sizes = size_or_sizes
        self.dim = dim

    def forward(self, X):
        return X.split(self.size_or_sizes, dim=self.dim)


# class Sum(nn.Module):
#     def forward(self, inps):
#         return sum(inps)

class Sum(nn.Module):
    def forward(self, inps):
        # 获取最大的第 2 和第 3 维度大小
        max_size_2 = max(x.size(2) for x in inps)
        max_size_3 = max(x.size(3) for x in inps)

        # 将所有张量扩展到最大的大小
        inps = [F.pad(x, (0, max_size_3 - x.size(3), 0, max_size_2 - x.size(2))) for x in inps]

        # 继续进行 sum 操作
        return sum(inps)


class AddAcross(nn.Module):
    def forward(self, inps):
        return [sum(items) for items in zip(*inps)]


class Reverse(nn.Module):
    def forward(self, inps):
        return inps[::-1]


class SelectOne(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        return xs[self.idx]


class Debug(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}')
        print(f'type: {type(x)}, len: {len(x)}')
        print(f'shapes: {self._get_shape_recurse(x)}')
        return x

    def _get_shape_recurse(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape
        return [self._get_shape_recurse(a) for a in x]
