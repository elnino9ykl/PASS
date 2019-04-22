#sqnet with special padding and upsampling, using 4 feature models and 1 fusion model

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as FFF
import numpy as np
import torch.optim as optim
import math
from sppad import Conv2d as spConv2d, UpsamplingBilinear2d

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes,use_sppad=False,split=1):
        super(Fire, self).__init__()
        #self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.conv1 = spConv2d(inplanes, squeeze_planes, kernel_size=1, stride=1,use_sppad=use_sppad,split=split)
        # self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ELU(inplace=True)
        #self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.conv2 = spConv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1,use_sppad=use_sppad,split=split)
        # self.bn2 = nn.BatchNorm2d(expand_planes)
        #self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = spConv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1,use_sppad=use_sppad,split=split)
        # self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ELU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        # out1 = self.bn2(out1)
        out2 = self.conv3(x)
        # out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ParallelDilatedConv(nn.Module):
    def __init__(self, inplanes, planes,use_sppad=False, split=1):
        super(ParallelDilatedConv, self).__init__()
        #self.dilated_conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1) 
        #self.dilated_conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2)
        #self.dilated_conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3)
        #self.dilated_conv_4 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dilated_conv_1 = spConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, use_sppad=use_sppad, split=split) 
        self.dilated_conv_2 = spConv2d(inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2, use_sppad=use_sppad,split=split)
        self.dilated_conv_3 = spConv2d(inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3, use_sppad=use_sppad,split=split)
        self.dilated_conv_4 = spConv2d(inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4, use_sppad=use_sppad,split=split)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

    def forward(self, x,only_encode=False):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out

class Net(nn.Module):
    def __init__(self, num_classes=28, encoder=None, use_sppad=True, split=4): #use special padding and upsampling, 4 feature models
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.split=split
         
        #self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1) # 32
        self.conv1 = spConv2d(3, 96, kernel_size=3, stride=2, padding=1, use_sppad=use_sppad, split=split) # 32
        # self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire1_1 = Fire(96, 16, 64, use_sppad=use_sppad, split=split)
        self.fire1_2 = Fire(128, 16, 64, use_sppad=use_sppad, split=split)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire2_1 = Fire(128, 32, 128, use_sppad=use_sppad, split=split)
        self.fire2_2 = Fire(256, 32, 128, use_sppad=use_sppad, split=split)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire3_1 = Fire(256, 64, 256, use_sppad=use_sppad, split=split)
        self.fire3_2 = Fire(512, 64, 256, use_sppad=use_sppad, split=split)
        self.fire3_3 = Fire(512, 64, 256, use_sppad=use_sppad, split=split)
        self.parallel = ParallelDilatedConv(512, 512, use_sppad=use_sppad, split=split)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ELU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ELU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 96, 3, stride=2, padding=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(96)
        self.relu4 = nn.ELU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(192, self.num_classes, 3, stride=2, padding=1, output_padding=1)

        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 32
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_1 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1) # 32
        self.conv1_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1) # 32

        #self.conv3_1 = spConv2d(256, 256, kernel_size=3, stride=1, padding=1, use_sppad=use_sppad, split=split) # 32
        #self.conv3_2 = spConv2d(512, 512, kernel_size=3, stride=1, padding=1, use_sppad=use_sppad, split=split) # 32
        #self.conv2_1 = spConv2d(128, 128, kernel_size=3, stride=1, padding=1, use_sppad=use_sppad, split=split) # 32
        #self.conv2_2 = spConv2d(256, 256, kernel_size=3, stride=1, padding=1, use_sppad=use_sppad, split=split) # 32
        #self.conv1_1 = spConv2d(96, 96, kernel_size=3, stride=1, padding=1, use_sppad=use_sppad, split=split) # 32
        #self.conv1_2 = spConv2d(192, 192, kernel_size=3, stride=1, padding=1, use_sppad=use_sppad, split=split) # 32

        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, only_encode=False):
        x = self.conv1(x)
        # x = self.bn1(x)
        x_1 = self.relu1(x)
        # print "x_1: %s" % str(x_1.size())
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        # print "x_2: %s" % str(x_2.size())
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        # print "x_3: %s" % str(x_3.size())
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)
        # print "x: %s" % str(x.size())

		###Merge for fusion model
        x_1 = FFF.max_pool2d(x_1, (1, self.split), (1, self.split)) #here
        x_2 = FFF.max_pool2d(x_2, (1, self.split), (1, self.split)) #here
        x_3 = FFF.max_pool2d(x_3, (1, self.split), (1, self.split)) #here
        x = FFF.max_pool2d(x, (1, self.split), (1, self.split)) #here

        y_3 = self.deconv1(x)
        y_3 = self.relu2(y_3)
        x_3 = self.conv3_1(x_3)
        x_3 = self.relu3_1(x_3)
        # print "y_3: %s" % str(y_3.size())
        # x = x.transpose(1, 2, 0)
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        # concat x_3
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        # print "y_2: %s" % str(y_2.size())
        # concat x_2
        x = torch.cat([x_2, y_2], 1)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        y_1 = self.deconv3(x)
        y_1 = self.relu4(y_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.relu1_1(x_1)
        # print "y_1: %s" % str(y_1.size())
        # concat x_1
        x = torch.cat([x_1, y_1], 1)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.deconv4(x)
        return x #, x_1, x_2, x_3, y_1, y_2, y_3
