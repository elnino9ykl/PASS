'''
@author hxx
@email anheidelonghu@gmail.com
@brief rewrite PyTorch Conv2d for special padding, the last dimension (W) of padding will be set to 0. the input will
       be divided into (split) blocks when processing.
@version    0.1, use_sppad and split is independent
'''
import torch
import torch.nn as nn
import collections
from itertools import repeat
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
from torch.nn.modules.pooling import _MaxPoolNd
from torch.nn.modules.upsampling import Upsample
import torch.nn.functional as F

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, use_sppad=False, split=0):

        self.split = split
        self.use_sppad = use_sppad
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if kernel_size[1] == 1:
            if self.use_sppad:
                print('kernel size at W is 1, use_sppad will be set to False')
            use_sppad = False
        elif kernel_size[1] == stride[1]:
            if self.use_sppad:
                print('kernel_size[1]=stride[1], use_sppad will be set to False')
            self.use_sppad = False
        elif padding[1] == 0:
            if use_sppad:
                print('padding[1] = 0, use_sppad will be set to False')
                self.use_sppad = False
        # elif self.split == 1:
        #     if use_sppad:
        #         print('split=1, use_sppad will be set to False')
        #         self.use_sppad = False

        if self.use_sppad:
            self.sppad = padding[1]
            padding = (padding[0], 0)

        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        # if not self.use_sppad:
        #     self,conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        #                           padding=padding, dilation=dilation, groups=groups, bias=bias)
        # else:
        #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        #                           padding=(padding, 0), dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if not self.use_sppad:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError(
                    'width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))
            blocks = []
            for i in range(self.split):
                block = input[..., width*i:width*(i+1)]
                block = F.conv2d(block, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
                blocks.append(block)
            return (torch.cat(blocks, 3))
        else:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError('width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))

            blocks = []
            left = input[...,-self.sppad:]
            right = input[...,:self.sppad]
            input = torch.cat([left, input, right], 3)
            for i in range(self.split):
                block = input[..., width * i:width * (i + 1) + 2 * self.sppad]
                block = F.conv2d(block, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
                blocks.append(block)

            return(torch.cat(blocks, 3))


# todo return_indices is not implemented
class MaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, use_sppad=False, split=2):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.use_sppad = use_sppad
        self.split = split
        if self.stride == self.kernel_size:
            self.use_sppad = False

        if self.use_sppad:
            self.sppad = self.padding[1]
            self.padding = (self.padding[0], 0)

    def forward(self, input):
        if self.use_sppad:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError('width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))
            blocks = []
            left = input[...,-self.sppad:]
            right = input[...,:self.sppad]
            input = torch.cat([left, input, right], 3)
            for i in range(self.split):
                block = input[..., width * i:width * (i + 1) + 2 * self.sppad]
                result = F.max_pool2d(block, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices) # be careful!!! return_indices is difficult!
                blocks.append(result)

            return(torch.cat(blocks, 3))
        else:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError(
                    'width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))
            blocks = []

            for i in range(self.split):
                block = input[..., width * i:width * (i + 1)]
                result = F.max_pool2d(block, self.kernel_size, self.stride,
                                      self.padding, self.dilation, self.ceil_mode,
                                      self.return_indices)  # be careful!!! return_indices is difficult!
                blocks.append(result)

            return (torch.cat(blocks, 3))

class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        if self.use_sppad:
            self.sppad = padding[1]
            padding = (padding[0], 0)

        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        if not self.use_sppad:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError(
                    'width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))

            blocks = []
            for i in range(self.split):
                block = input[..., width * i:width * (i + 1)]
                block = F.conv_transpose2d(
                    block, self.weight, self.bias, self.stride, self.padding,
                    output_padding, self.groups, self.dilation)
                blocks.append(block)

            return (torch.cat(blocks, 3))
        else:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError('width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))

            blocks = []
            left = input[...,-self.sppad:]
            right = input[...,:self.sppad]
            input = torch.cat([left, input, right], 3)
            for i in range(self.split):
                block = input[..., width * i:width * (i + 1) + 2 * self.sppad]
                block = F.conv_transpose2d(
                block, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
                blocks.append(block)

            return(torch.cat(blocks, 3))

class UpsamplingBilinear2d(Upsample):
    def __init__(self, size=None, scale_factor=None, use_sppad=False, split=2):
        super(UpsamplingBilinear2d, self).__init__(size, scale_factor, mode='bilinear', align_corners=True)
        self.use_sppad = use_sppad
        self.split = split
        self.sppad = 1 #todo if 1 is enhough?

    def forward(self, input):
        if not self.use_sppad:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError(
                    'width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))

            blocks = []
            for i in range(self.split):
                block = input[..., width * i:width * (i + 1)]
                block = super(UpsamplingBilinear2d, self).forward(block)
                # block = block[..., 1 * self.scale_factor:-1 * self.scale_factor]
                blocks.append(block)

            return (torch.cat(blocks, 3))
        else:
            width = input.shape[3] // self.split
            if width * self.split != input.shape[3]:
                raise NotImplementedError(
                    'width {} can not be divided into {} block'.format(str(input.shape[3]), str(self.split)))

            blocks = []
            left = input[..., -self.sppad:]
            right = input[..., :self.sppad]
            input = torch.cat([left, input, right], 3)
            for i in range(self.split):
                block = input[..., width * i:width * (i + 1) + 2 * self.sppad]
                block = super(UpsamplingBilinear2d, self).forward(block)
                block = block[..., 1 * self.scale_factor:-1 * self.scale_factor]
                blocks.append(block)

            return (torch.cat(blocks, 3))
