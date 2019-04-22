# ERF-APSPNet full model definition for Pytorch 
# Separated into 4 feature models and 1 fusion model, with special padding and upsampling operations
# April 2019
# Kailun Yang
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=(1, 0), bias=True)  # change padding
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        width = input.shape[3] // 4
        blockA = input[:,:,:,:width]
        blockB = input[:,:,:,width:width*2]
        blockC = input[:,:,:,width*2:width*3]
        blockD = input[:,:,:,-width:]

        poolA = self.pool(blockA)
        poolB = self.pool(blockB)
        poolC = self.pool(blockC)
        poolD = self.pool(blockD)

        leftA = blockD[:, :, :, -1:]
        rightA = blockB[:, :, :, :1]
        leftB = blockA[:, :, :, -1:]
        rightB = blockC[:, :, :, :1]
        leftC = blockB[:, :, :, -1:]
        rightC = blockD[:, :, :, :1]
        leftD = blockC[:, :, :, -1:]
        rightD = blockA[:, :, :, :1]

        blockA = torch.cat([leftA, blockA, rightA], 3)  # change input
        blockB = torch.cat([leftB, blockB, rightB], 3)
        blockC = torch.cat([leftC, blockC, rightC], 3)
        blockD = torch.cat([leftD, blockD, rightD], 3)

        outputA = self.conv(blockA)
        outputB = self.conv(blockB)
        outputC = self.conv(blockC)
        outputD = self.conv(blockD)

        outputA = torch.cat([outputA, poolA], 1)
        outputB = torch.cat([outputB, poolB], 1)
        outputC = torch.cat([outputC, poolC], 1)
        outputD = torch.cat([outputD, poolD], 1)

        output = torch.cat([outputA, outputB, outputC, outputD], 3)
        output = self.bn(output)

        return F.relu(output)

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)  # no change

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 0), bias=True)  # change padding

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))  # no change

        self.dilated = 1 * dilated
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 0), bias=True,
                                   dilation=(1, dilated))  # change padding

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        width = input.shape[3] // 4
        blockA = input[:,:,:,:width]
        blockB = input[:,:,:,width:width*2]
        blockC = input[:,:,:,width*2:width*3]
        blockD = input[:,:,:,-width:]

        outputA = self.conv3x1_1(blockA)
        outputA = F.relu(outputA)
        outputB = self.conv3x1_1(blockB)
        outputB = F.relu(outputB)
        outputC = self.conv3x1_1(blockC)
        outputC = F.relu(outputC)
        outputD = self.conv3x1_1(blockD)
        outputD = F.relu(outputD)

        leftA = outputD[:, :, :, -1:]
        rightA = outputB[:, :, :, :1]
        leftB = outputA[:, :, :, -1:]
        rightB = outputC[:, :, :, :1]
        leftC = outputB[:, :, :, -1:]
        rightC = outputD[:, :, :, :1]
        leftD = outputC[:, :, :, -1:]
        rightD = outputA[:, :, :, :1]

        outputA = torch.cat([leftA, outputA, rightA], 3)
        outputB = torch.cat([leftB, outputB, rightB], 3)
        outputC = torch.cat([leftC, outputC, rightC], 3)
        outputD = torch.cat([leftD, outputD, rightD], 3) 
        outputA = self.conv1x3_1(outputA)  # hxx
        outputB = self.conv1x3_1(outputB)  # hxx
        outputC = self.conv1x3_1(outputC)  # hxx
        outputD = self.conv1x3_1(outputD)  # hxx

        output = torch.cat([outputA, outputB, outputC, outputD], 3)
        output = self.bn1(output)
        output = F.relu(output)

        hwidth = output.shape[3] // 4
        outputA = output[:,:,:,:hwidth]
        outputB = output[:,:,:,hwidth:hwidth*2]
        outputC = output[:,:,:,hwidth*2:hwidth*3]
        outputD = output[:,:,:,-hwidth:]

        outputA = self.conv3x1_2(outputA)
        outputA = F.relu(outputA)
        outputB = self.conv3x1_2(outputB)
        outputB = F.relu(outputB)
        outputC = self.conv3x1_2(outputC)
        outputC = F.relu(outputC)
        outputD = self.conv3x1_2(outputD)
        outputD = F.relu(outputD)

        leftA = outputD[:, :, :, -self.dilated:]
        rightA = outputB[:, :, :, :self.dilated]
        leftB = outputA[:, :, :, -self.dilated:]
        rightB = outputC[:, :, :, :self.dilated]
        leftC = outputB[:, :, :, -self.dilated:]
        rightC = outputD[:, :, :, :self.dilated]
        leftD = outputC[:, :, :, -self.dilated:]
        rightD = outputA[:, :, :, :self.dilated]

        outputA = torch.cat([leftA, outputA, rightA], 3)
        outputB = torch.cat([leftB, outputB, rightB], 3)
        outputC = torch.cat([leftC, outputC, rightC], 3)
        outputD = torch.cat([leftD, outputD, rightD], 3)
        outputA = self.conv1x3_2(outputA)  # hxx
        outputB = self.conv1x3_2(outputB)  # hxx
        outputC = self.conv1x3_2(outputC)  # hxx
        outputD = self.conv1x3_2(outputD)  # hxx

        output = torch.cat([outputA, outputB, outputC, outputD], 3)
        output = self.bn2(output)
        # output = F.relu(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
           
        return F.relu(output + input)  # +input = identity (residual connection)

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.00, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.0, 2))
            self.layers.append(non_bottleneck_1d(128, 0.0, 4))
            self.layers.append(non_bottleneck_1d(128, 0.0, 8))
            self.layers.append(non_bottleneck_1d(128, 0.0, 16))

        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)  # no use, no change

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)
        
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=(1, 0), output_padding=(1, 0),
                                       bias=True)  # change two padding
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        left = input[:, :, :, -1:]
        right = input[:, :, :, :1]
        input = torch.cat([left, input, right], 3)  # change input
        output = self.conv(input)  # hxx
        output = output[:, :, :, 1:-1]
        output = self.bn(output)
        return F.relu(output)

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the feature
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PSPDec(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=(64, 128), use_new_padding=False):
        super(PSPDec, self).__init__()
        self.downsize = downsize
        self.use_new_padding = False#use_new_padding
        self.features = nn.Sequential(
            nn.AvgPool2d(downsize, stride=downsize),
            nn.Conv2d(in_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features, momentum=.95),
            nn.ReLU(inplace=True),
        )
        self.F_glo=FGlo(in_features,reduction=16)
        
        if self.use_new_padding:
            if downsize[1] == 128:
                self.upsample = nn.Upsample(size=upsize, mode='bilinear')
            else:
                self.upsample = nn.Upsample(size=(upsize[0], upsize[1] + downsize[1] * 2), mode='bilinear')
        else:
            self.upsample = nn.Upsample(size=upsize, mode='bilinear')

    def forward(self, x):
        width = x.shape[3] // 4
        blockA = x[:,:,:,:width]
        blockB = x[:,:,:,width:width*2]
        blockC = x[:,:,:,width*2:width*3]
        blockD = x[:,:,:,-width:]
        
        xA = self.F_glo(blockA)
        xB = self.F_glo(blockB)
        xC = self.F_glo(blockC)
        xD = self.F_glo(blockD)

        xA = self.features(xA)
        xB = self.features(xB)
        xC = self.features(xC)
        xD = self.features(xD)

        if self.use_new_padding:
            if self.downsize[1] == 128:
                xA = self.upsample(xA)
                xB = self.upsample(xB)
                xC = self.upsample(xC)
                xD = self.upsample(xD)
            else:
                leftA = xD[:, :, :, -1:]
                rightA = xB[:, :, :, :1]
                leftB = xA[:, :, :, -1:]
                rightB = xC[:, :, :, :1]
                leftC = xB[:, :, :, -1:]
                rightC = xD[:, :, :, :1]
                leftD = xC[:, :, :, -1:]
                rightD = xA[:, :, :, :1]

                xA = torch.cat([leftA, xA, rightA], 3)
                xA = self.upsample(xA)
                xA = xA[:, :, :, self.downsize[1]:-self.downsize[1]]
                xB = torch.cat([leftB, xB, rightB], 3)
                xB = self.upsample(xB)
                xB = xB[:, :, :, self.downsize[1]:-self.downsize[1]]
                xC = torch.cat([leftC, xC, rightC], 3)
                xC = self.upsample(xC)
                xC = xC[:, :, :, self.downsize[1]:-self.downsize[1]]
                xD = torch.cat([leftD, xD, rightD], 3)
                xD = self.upsample(xC)
                xD = xD[:, :, :, self.downsize[1]:-self.downsize[1]]
        else:
            xA = self.upsample(xA)
            xB = self.upsample(xB)
            xC = self.upsample(xC)
            xD = self.upsample(xD)
        x = torch.cat([xA, xB, xC, xD], 3)
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes, use_new_padding=False):
        super().__init__()
        self.use_new_padding = use_new_padding
       
        self.layer5a = PSPDec(128, 32, (64, 128), (64, 128), use_new_padding=self.use_new_padding)
        self.layer5b = PSPDec(128, 32, (32, 64), (64, 128), use_new_padding=self.use_new_padding)
        self.layer5c = PSPDec(128, 32, (16, 32), (64, 128), use_new_padding=self.use_new_padding)
        self.layer5d = PSPDec(128, 32, (8, 16), (64, 128), use_new_padding=self.use_new_padding)

        self.final = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False), # todo pretty important
            nn.BatchNorm2d(256, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Conv2d(256, num_classes, 1),
        )
        self.pooling = nn.MaxPool2d((1,4))

    def forward(self, x):
        
        
        x = torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1)
        
        x = self.pooling(x)
        x = self.final(x)
     
        if False:
            leftA = xD[:,:,:,-1:]
            rightA = xB[:,:,:,:1]
            xA = torch.cat([leftA,xA,rightA],3)
            leftB = xA[:,:,:,-1:]
            rightB = xC[:,:,:,:1]
            xB = torch.cat([leftB,xB,rightB],3)
            leftC = xB[:,:,:,-1:]
            rightC = xD[:,:,:,:1]
            xC = torch.cat([leftC,xC,rightC],3)
            leftD = xC[:,:,:,-1:]
            rightD = xA[:,:,:,:1]
            xD = torch.cat([leftD,xD,rightD],3)
            xA = F.upsample(xA, size=(512, 1024 + 8 * 2), mode='bilinear')
            xA = xA[:, :, :, 8:-8]
            xB = F.upsample(xB, size=(512, 1024 + 8 * 2), mode='bilinear')
            xB = xB[:, :, :, 8:-8]
            xC = F.upsample(xC, size=(512, 1024 + 8 * 2), mode='bilinear')
            xC = xC[:, :, :, 8:-8]
            xD = F.upsample(xD, size=(512, 1024 + 8 * 2), mode='bilinear')
            xD = xD[:, :, :, 8:-8]
      
        if self.use_new_padding:
            left = x[:,:,:,-1:]
            right = x[:,:,:,:1]
            x = torch.cat([left,x,right],3)
            x = F.upsample(x,size=(512,1024+8*2), mode='bilinear')
            x = x[:,:,:,8:-8]

        return F.upsample(x, size=(692, 2048), mode='bilinear')


class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder

        self.decoder = Decoder(num_classes, use_new_padding=True)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)
