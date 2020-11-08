# -*- coding: utf-8 -*-

# @Time    : 2020/11/4 16:04
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numbers
from typing import Union, List, Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch import multiply
from torch import nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter

class Indexes1xNoWeight(Module):
    _shape_t = Union[int, List[int], Tuple, Size]
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    elementwise_affine: bool

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape: _shape_t, axis: int = 0, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.indexes_axis = axis
        self.index_tensor = None
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        self.indexes_shape = tuple(shape)
        self.generate_tensor(shape)
        self.elementwise_affine = elementwise_affine


    def generate_tensor(self, shape: tuple) -> None:

        index_tensor = torch.arange(1, shape[self.indexes_axis] + 1)
        for i in range(len(shape)):
            if self.indexes_axis != i:
                index_tensor = index_tensor.unsqueeze(i)

        index_tensor2 = torch.ones(shape)
        self.index_tensor = torch.multiply(index_tensor, index_tensor2)
        self.index_tensor = self.index_tensor.unsqueeze(0)
        self.index_tensor = self.index_tensor.unsqueeze(0)
        self.index_tensor.requires_grad = True

    def forward(self, input: Tensor) -> Tensor:
        return multiply(input, self.index_tensor.to(input.device))+1e-8
        # self.generate_tensor(self, input.shape)
        # return multiply(multiply(input, self.weight), self.index_tensor)

    def extra_repr(self) -> Tensor:
        return 'In={in_channels}, Out={in_channels},axis={indexes_axis}, {indexes_shape}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class IndexesNoWeight(Indexes1xNoWeight):
    _shape_t = Union[int, List[int], Tuple, Size]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape: _shape_t, elementwise_affine: bool = True) -> None:
        super().__init__(in_channels,
                         out_channels,
                         shape, 0, elementwise_affine, )

    def generate_tensor(self, shape: tuple) -> None:

        ch1 = self.out_channels // len(shape)
        ch2 = self.out_channels - len(shape) * ch1 + ch1
        its = []
        for n, i in enumerate(shape):
            index_tensor = torch.arange(1, i + 1)
            for i in range(len(shape)):
                if n != i:
                    index_tensor = index_tensor.unsqueeze(i)

            index_tensor2 = torch.ones(shape)
            index_tensor = torch.multiply(index_tensor, index_tensor2)
            if n == 0:
                ch_ = ch2
            else:
                ch_ = ch1
            index_tensor = index_tensor.unsqueeze(0)
            index_tensor = index_tensor.unsqueeze(0)
            index_tensor = index_tensor.repeat_interleave(ch_, dim=1)
            its.append(index_tensor)

        self.index_tensor = torch.cat(its, dim=1)
        self.index_tensor.requires_grad = True

    def extra_repr(self) -> Tensor:
        return 'In={in_channels}, Out={in_channels}, {indexes_shape}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Indexes1x(Module):
    _shape_t = Union[int, List[int], Tuple, Size]
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    elementwise_affine: bool

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape: _shape_t, axis: int = 0, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.indexes_axis = axis
        self.index_tensor = None
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        self.indexes_shape = tuple(shape)
        self.generate_tensor(shape)

        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels, *shape))
            self.bias = Parameter(torch.Tensor(out_channels))

        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def generate_tensor(self, shape: tuple) -> None:

        index_tensor = torch.arange(1, shape[self.indexes_axis] + 1)
        for i in range(len(shape)):
            if self.indexes_axis != i:
                index_tensor = index_tensor.unsqueeze(i)

        index_tensor2 = torch.ones(shape)
        self.index_tensor = torch.multiply(index_tensor, index_tensor2)
        self.index_tensor = self.index_tensor.unsqueeze(0)
        self.index_tensor = self.index_tensor.unsqueeze(0)
        self.index_tensor.requires_grad = True

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return multiply(multiply(input, self.weight), self.index_tensor.to(input.device))+1e-8
        # self.generate_tensor(self, input.shape)
        # return multiply(multiply(input, self.weight), self.index_tensor)

    def extra_repr(self) -> Tensor:
        return 'In={in_channels}, Out={in_channels},axis={indexes_axis}, {indexes_shape}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Indexes(Indexes1x):
    _shape_t = Union[int, List[int], Tuple, Size]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape: _shape_t, elementwise_affine: bool = True) -> None:
        super().__init__(in_channels,
                         out_channels,
                         shape, 0, elementwise_affine, )

    def generate_tensor(self, shape: tuple) -> None:

        ch1 = self.out_channels // len(shape)
        ch2 = self.out_channels - len(shape) * ch1 + ch1
        its = []
        for n, i in enumerate(shape):
            index_tensor = torch.arange(1, i + 1)
            for i in range(len(shape)):
                if n != i:
                    index_tensor = index_tensor.unsqueeze(i)

            index_tensor2 = torch.ones(shape)
            index_tensor = torch.multiply(index_tensor, index_tensor2)
            if n == 0:
                ch_ = ch2
            else:
                ch_ = ch1
            index_tensor = index_tensor.unsqueeze(0)
            index_tensor = index_tensor.unsqueeze(0)
            index_tensor = index_tensor.repeat_interleave(ch_, dim=1)
            its.append(index_tensor)

        self.index_tensor = torch.cat(its, dim=1)
        self.index_tensor.requires_grad = True

    def extra_repr(self) -> Tensor:
        return 'In={in_channels}, Out={in_channels}, {indexes_shape}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


if __name__ == "__main__":
    # ins = Indexes(in_channels=1, out_channels=6, shape=(6, 5, 4))
    ins = Indexes1x(in_channels=1, out_channels=6, shape=(6, 5, 4), axis=0)
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    ins.to(device)
    a = torch.randn((100, 1, 6, 5, 4))
    a = a.to(device)
    b = ins(a)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(b, b + 1.0)

    loss.backward()
