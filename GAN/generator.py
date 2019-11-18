"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class generator(nn.Module):

    def __init__(self):
        super(generator,self).__init__()
        self.fc1 = nn.Linear(100,196*4*4)

        self._conv_layers = nn.Sequential(OrderedDict([
          ('conv1', nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)),
          ('BN1', nn.BatchNorm2d(196)),
          ('relu1', nn.ReLU(inplace = True)),

          ('conv2', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('BN2', nn.BatchNorm2d(196)),
          ('relu2', nn.ReLU(inplace = True)),

          ('conv3', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('BN3', nn.BatchNorm2d(196)),
          ('relu3', nn.ReLU(inplace = True)),

          ('conv4', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('BN4', nn.BatchNorm2d(196)),
          ('relu4', nn.ReLU(inplace = True)),

          ('conv5', nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)),
          ('BN5', nn.BatchNorm2d(196)),
          ('relu5', nn.ReLU(inplace = True)),

          ('conv6', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('BN6', nn.BatchNorm2d(196)),
          ('relu6', nn.ReLU(inplace = True)),

          ('conv7', nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)),
          ('BN7', nn.BatchNorm2d(196)),
          ('relu7', nn.ReLU(inplace = True)),

          ('conv8', nn.Conv2d(in_channels = 196, out_channels = 3, kernel_size = 3, padding = 1, stride = 1)),
          ('tanh8', nn.Tanh()),
        ]))

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1,196,4,4)
        ret = self._conv_layers(x)
        return ret
