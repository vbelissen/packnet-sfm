# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth

########################################################################################################################

class DepthCAMResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.cam_convs = CamConvs()
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def _concat_features(self, enc_features, cam_features):
        features = []
        n = len(enc_features)
        for i in range(n):
            features.append(torch.cat([enc_features[i],
                                       F.interpolate(cam_features,
                                                     size=enc_features[i].shape, # a corriger
                                                     mode='bilinear',
                                                     align_corners=True)
                                       ], 1))
        return(features)

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        c = self.cam_convs(x) # (B, C, H, W) avec C le nombre de channels des cam convs
        x = self.encoder(x)  # [(B, 64, H/2, W/2), (B, 64, H/4, W/4), (B, 128, H/8, W/8), (B, 256, H/16, W/16), (B, 512, H/32, W/32)]
        x_c = self._concat_features(x, c)
        x = self.decoder(x_c)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

########################################################################################################################
