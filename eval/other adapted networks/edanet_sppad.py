#######################
# name: EDANet full model definition reproduced by Pytorch(v0.4.1)
# EDANet with special padding and upsampling, using 4 feature models and 1 fusion model
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sppad import Conv2d as spConv2d, UpsamplingBilinear2d

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, use_sppad=False, split=1):
        super(DownsamplerBlock,self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = spConv2d(ninput, noutput-ninput, kernel_size=3, stride=2, padding=1, use_sppad=use_sppad, split=split)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = spConv2d(ninput, noutput, kernel_size=3, stride=2, padding=1, use_sppad=use_sppad, split=split)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)
    

class EDABlock(nn.Module):
    def __init__(self,ninput, dilated, k = 40,dropprob = 0.02, use_sppad=False, split=1):
        super(EDABlock,self).__init__()

        #k: growthrate
        #dropprob:a dropout layer between the last ReLU and the concatenation of each module

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1),padding=(1,0))
        self.conv1x3_1 = spConv2d(k, k, kernel_size=(1, 3),padding=(0,1), use_sppad=use_sppad, split=split)
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated,0), dilation = dilated)
        self.conv1x3_2 = spConv2d(k, k, (1,3), stride=1, padding=(0,dilated), dilation =  dilated, use_sppad=use_sppad, split=split)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        output = torch.cat([output,input],1)
        #print output.size() #check the output
        return output


class Net(nn.Module):
    def __init__(self, num_classes=28, use_sppad=True, split=4): #use special padding and upsampling, 4 feature models
        super(Net,self).__init__()
        self.use_sppad = use_sppad
        self.split = split

        self.layers = nn.ModuleList()
        self.dilation1 = [1,1,1,2,2]
        self.dilation2 = [2,2,4,4,8,8,16,16]

        # DownsamplerBlock1
        self.layers.append(DownsamplerBlock(3, 15, use_sppad=use_sppad, split=split))

        # DownsamplerBlock2
        self.layers.append(DownsamplerBlock(15, 60, use_sppad=use_sppad, split=split))

        # EDA module 1-1 ~ 1-5
        for i in range(5):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i], use_sppad=use_sppad, split=split))

        # DownsamplerBlock3
        self.layers.append(DownsamplerBlock(260, 130, use_sppad=use_sppad, split=split))

        # EDA module 2-1 ~ 2-8
        for j in range(8):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j], use_sppad=use_sppad, split=split))

        # Projection layer
        self.project_layer = nn.Conv2d(450,num_classes,kernel_size = 1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x):

        output = x

        for layer in self.layers:
            output = layer(output)

        output = F.max_pool2d(output, (1, self.split), (1, self.split)) #here, merger for fusion model
        output = self.project_layer(output)

        # Bilinear interpolation x8
        # output = F.interpolate(output,scale_factor = 8,mode = 'bilinear',align_corners=True)
        output = UpsamplingBilinear2d(scale_factor=8, use_sppad=self.use_sppad, split=1)(output)


        return output
