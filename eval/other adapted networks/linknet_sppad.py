#LinkNet with special padding and upsampling, using 4 feature models and 1 fusion model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sppad import Conv2d as spConv2d, UpsamplingBilinear2d

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False, use_sppad=False, split=1):
        super(BasicBlock, self).__init__()
        self.conv1 = spConv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias, use_sppad=use_sppad, split=split)
        #self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spConv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias, use_sppad=use_sppad, split=split)
        #self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(spConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, use_sppad=use_sppad, split=split),
                            nn.BatchNorm2d(out_planes),)
            #self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            #nn.BatchNorm2d(out_planes),)
            

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False, use_sppad=False, split=1):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, use_sppad=use_sppad, split=split)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, use_sppad=use_sppad, split=split)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class Net(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, n_classes=28,encoder=None, use_sppad=True, split=4): #use special padding and upsampling, 4 feature models
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(Net, self).__init__()
        self.split=split
        #self.use_sppad=use_sppad
        self.conv1 = spConv2d(3, 64, 7, 2, 3, bias=False, use_sppad=use_sppad, split=split)
        #self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1, use_sppad=use_sppad, split=split)
        self.encoder2 = Encoder(64, 128, 3, 2, 1, use_sppad=use_sppad, split=split)
        self.encoder3 = Encoder(128, 256, 3, 2, 1, use_sppad=use_sppad, split=split)
        self.encoder4 = Encoder(256, 512, 3, 2, 1, use_sppad=use_sppad, split=split)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax()

    def forward(self, x, only_encode=False):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

		#here, merge for fusion model
        x = F.max_pool2d(x, (1, self.split), (1, self.split)) #here
        e1 = F.max_pool2d(e1, (1, self.split), (1, self.split)) #here
        e2 = F.max_pool2d(e2, (1, self.split), (1, self.split)) #here
        e3 = F.max_pool2d(e3, (1, self.split), (1, self.split)) #here
        e4 = F.max_pool2d(e4, (1, self.split), (1, self.split)) #here

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y
