
from __future__ import print_function, division
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class FmConvNet(nn.Module):
    def __init__(self, num_cls=3, in_channels=256, out_channels=2048, in_size=(7,7), pretrain=False):
        super(FmConvNet, self).__init__()
        self.num_cls = num_cls
        #"""
        self.feature_extractor = nn.Sequential(
                                    nn.Conv2d(in_channels, 512, 3, padding=2),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 512, 1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 1024, 3),
                                    nn.BatchNorm2d(1024),
                                    nn.Conv2d(1024, 1024, 3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.Conv2d(1024, 1024, 1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, out_channels, 3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Conv2d(out_channels, out_channels, 1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)
                                 )
        """
        self.feature_extractor = nn.Sequential(nn.Conv2d(256, 512, 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 128, 3, padding=1),
                  nn.BatchNorm2d(128),
                  nn.Conv2d(128, 512, 1),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True))
        """
        self.pool = nn.AvgPool2d(in_size, stride=1)
        self.fc = nn.Linear(out_channels, num_cls)
        self.out_channels = out_channels

    def forward(self, x):
        attn_features = self.feature_extractor(x)
        after_pool = self.pool(attn_features)
        x = after_pool.view((after_pool.size()[0], -1))
        x = self.fc(x)
        return x, attn_features, after_pool
    
    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))
