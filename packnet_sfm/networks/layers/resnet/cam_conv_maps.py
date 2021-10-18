
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn



class CamConvMaps(nn.Module):
    def __init__(self):
        super(CamConvMaps, self).__init__()
        self.num_maps = 7

    def forward(self, input_features):
        B, C, H, W = input_features.shape
        device = input_features.get_device()
        output = torch.zeros(B, self.num_maps, H, W).to(device) # test
        return output
