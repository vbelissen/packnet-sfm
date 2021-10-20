# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from packnet_sfm.networks.layers.resnet.cam_conv_maps import CamConvMaps
from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.pose_decoder import PoseDecoder

########################################################################################################################

class PoseCAMResNet(nn.Module):
    """
    Pose network based on the ResNet architecture.

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
        assert version is not None, "PoseResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        #self.cam_convs = CamConvMaps()
        # self.decoder = PoseDecoder(self.encoder.num_ch_enc + 2 * self.cam_convs.num_maps,
        #                            num_input_features=1,
        #                            num_frames_to_predict_for=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc + 2 * 7,
                                   num_input_features=1,
                                   num_frames_to_predict_for=2)

    def _concat_2_features(self, enc_features, cam_features_1, cam_features_2):
        """
        H, W : size of input images
        enc_features: [(B, 64, H/2, W/2), (B, 64, H/4, W/4), (B, 128, H/8, W/8), (B, 256, H/16, W/16), (B, 512, H/32, W/32)]
        cam_features: (B, Ccam, H, W)

        output size:
        [(B, 64 + Ccam, H/2, W/2), (B, 64 + Ccam, H/4, W/4), (B, 128 + Ccam, H/8, W/8), (B, 256 + Ccam, H/16, W/16), (B, 512 + Ccam, H/32, W/32)]
        """
        # TODO :
        # En réalité seule la dernière feature map est utilisée par le décodeur
        # résolution B x 512 x H/32 x W/32
        # => pas forcément obliger d'interpoler/concaténer sur chaque niveau
        features = []
        n = len(enc_features)
        Bcam, Ccam, Hcam, Wcam = cam_features_1.shape
        for i in range(n):
            Benc_i, Cenc_i, Henc_i, Wenc_i = enc_features[i].shape
            assert Bcam == Benc_i
            features.append(torch.cat([enc_features[i],
                                       F.interpolate(cam_features_1,
                                                     size=(Henc_i, Wenc_i),
                                                     mode='bilinear',
                                                     align_corners=True),
                                       F.interpolate(cam_features_2,
                                                     size=(Henc_i, Wenc_i),
                                                     mode='bilinear',
                                                     align_corners=True)
                                       ], 1))
        return (features)

    def forward(self, target_image, cam_features, ref_imgs, ref_cam_features):
        """
        Runs the network and returns predicted poses
        (1 for each reference image).
        """
        outputs = []
        #target_cam_conv = self.cam_convs(target_image)
        #ref_cam_convs = [self.cam_convs(ref_img) for ref_img in ref_imgs]
        for i, ref_img in enumerate(ref_imgs):
            inputs = torch.cat([target_image, ref_img], 1)
            axisangle, translation = self.decoder([self._concat_2_features(self.encoder(inputs), cam_features, ref_cam_features[i])])
            outputs.append(torch.cat([translation[:, 0], axisangle[:, 0]], 2))
        pose = torch.cat(outputs, 1)
        return pose

########################################################################################################################

