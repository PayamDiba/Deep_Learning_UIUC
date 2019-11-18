"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class critic(nn.Module):

    def __init__(self):
        super(critic,self).__init__()

        self._conv_layers = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(in_channels = 3, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('LN1', nn.LayerNorm([196,32,32])),
          ('relu1', nn.LeakyReLU(inplace = True)),

          ('conv2', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 2)),
          ('LN2', nn.LayerNorm([196,16,16])),
          ('relu2', nn.LeakyReLU(inplace = True)),

          ('conv3', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('LN3', nn.LayerNorm([196,16,16])),
          ('relu3', nn.LeakyReLU(inplace = True)),

          ('conv4', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 2)),
          ('LN4', nn.LayerNorm([196,8,8])),
          ('relu4', nn.LeakyReLU(inplace = True)),

          ('conv5', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('LN5', nn.LayerNorm([196,8,8])),
          ('relu5', nn.LeakyReLU(inplace = True)),

          ('conv6', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('LN6', nn.LayerNorm([196,8,8])),
          ('relu6', nn.LeakyReLU(inplace = True)),

          ('conv7', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)),
          ('LN7', nn.LayerNorm([196,8,8])),
          ('relu7', nn.LeakyReLU(inplace = True)),

          ('conv8', nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 2)),
          ('LN8', nn.LayerNorm([196,4,4])),
          ('relu8', nn.LeakyReLU(inplace = True)),
          ('pool', nn.MaxPool2d(kernel_size=4, stride=4))
        ]))

        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)

    def forward(self, x, extract_features=0):
        if extract_features == 4:
            self._conv_layers2 = nn.Sequential(*list(self._conv_layers.children())[:12])
            x = self._conv_layers2(x)
            x = F.max_pool2d(x,8,8)
            x = x.view(-1,196)
            return x
            
        x = self._conv_layers(x)
        x = x.view(-1,196)
        if extract_features == 8:
            return x
        ret1 = self.fc1(x)
        ret2 = self.fc10(x)
        return ret1, ret2
