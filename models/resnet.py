
from __future__ import print_function, division
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

LOCAL_WEIGHT_PATH = "./model_weights/init_resnet/resnet50.pth"

class Resnet50(nn.Module):
    def __init__(self, num_cls=3, in_channels=3, \
                    in_size=(224,224), out_channels=2048, \
                    use_rgb=True, pretrain=True, load_local=False):
        super(Resnet50, self).__init__()
        self.num_cls = num_cls
        # get the pretrained Resnet50 network
        self.net = models.resnet50(pretrained=pretrain)
        # placeholder for the gradients
        num_ftrs = self.net.fc.in_features
        self.out_channels = num_ftrs
        self.net.fc = nn.Linear(num_ftrs, num_cls)
        if load_local:
            self.load_model(LOCAL_WEIGHT_PATH)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        attn_features = self.net.layer4(x)
        after_pool = self.net.avgpool(attn_features)
        x = after_pool.view((after_pool.size()[0], -1))
        x = self.net.fc(x)
        return x, attn_features, after_pool
    
    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))
