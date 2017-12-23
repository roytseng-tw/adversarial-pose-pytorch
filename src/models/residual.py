"""
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
Roy Tseng, 2017.11.23
"""
import torch.nn as nn


class HgResBlock(nn.Module):
    ''' Hourglass residual block '''
    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1, stride)  # bias=False
        self.bn2 = nn.BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(midplanes, midplanes, 3, stride, 1)
        self.bn3 = nn.BatchNorm2d(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, 1, stride)  # bias=False
        self.relu = nn.ReLU(inplace=True)
        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, 1, 1)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual
        return out
