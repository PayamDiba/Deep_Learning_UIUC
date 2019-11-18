"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv = self._conv_layers()
        self.fc = self._fc_layers()


    def _conv_layers(self):
        ret = nn.Sequential(OrderedDict([
          ('conv1_1', nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)),
          ('batchnorm1_1', nn.BatchNorm2d(32)),
          ('relu1_1', nn.ReLU(inplace = True)),
          ('conv1_2', nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)),
          ('batchnorm1_2', nn.BatchNorm2d(32)),
          ('relu1_2', nn.ReLU(inplace = True)),
          ('maxpool1_1', nn.MaxPool2d(kernel_size=2, stride=2)),
          ('dropout1', nn.Dropout2d(p = 0.05)),

          ('conv2_1', nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)),
          ('batchnorm2_1', nn.BatchNorm2d(64)),
          ('relu2_1', nn.ReLU(inplace = True)),
          ('conv2_2', nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)),
          ('batchnorm2_2', nn.BatchNorm2d(64)),
          ('relu2_2', nn.ReLU(inplace = True)),
          ('maxpool2_1', nn.MaxPool2d(kernel_size=2, stride=2)),
          ('dropout2', nn.Dropout2d(p = 0.05)),

          ('conv3_1', nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)),
          ('batchnorm3_1', nn.BatchNorm2d(128)),
          ('relu3_1', nn.ReLU(inplace = True)),
          ('conv3_2', nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)),
          ('batchnorm3_2', nn.BatchNorm2d(128)),
          ('relu3_2', nn.ReLU(inplace = True)),
          ('conv3_3', nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)),
          ('batchnorm3_3', nn.BatchNorm2d(128)),
          ('relu3_3', nn.ReLU(inplace = True)),
          ('maxpool3_1', nn.MaxPool2d(kernel_size=2, stride=2)),

        ]))

        return ret

    def _fc_layers(self):
        ret = nn.Sequential(OrderedDict([
          ('dropout2', nn.Dropout(p = 0.05)),
          ('fc1', nn.Linear(2048,512)),
          ('relu4_1', nn.ReLU(inplace = True)),
          ('fc2', nn.Linear(512,128)),
          ('relu4_2', nn.ReLU(inplace = True)),
          ('last', nn.Linear(128,10)),

        ]))

        return ret

    def forward(self, x):

        conv_out = self.conv(x)
        conv_out_flatten = conv_out.view(conv_out.size(0),-1)
        forw = self.fc(conv_out_flatten)

        return forw
