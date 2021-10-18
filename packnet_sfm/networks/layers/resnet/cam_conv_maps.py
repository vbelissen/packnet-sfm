
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn



class CamConvMaps(nn.Module):
    def __init__(self):
        super(CamConvMaps, self).__init__()

    def forward(self, input_features):
        B, C, H, W = input_features.shape
        output = torch.zeros(B, 7, H, W) # test
        return output
