import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50


class ResnetFPN(nn.Module):
    def __init__(self, n_labels):
        super(ResnetFPN, self).__init__()
        self.resnet_backbone = resnet50()
        self._up_kwargs = {"mode": "bilinear", "align_corners": False}
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classify = nn.Conv2d(128 * 4, n_labels, kernel_size=3, stride=1, padding=1)

        self.pre_rotate = np.array([0, 90, 180, 270]) * np.pi / 180
        self.post_rotate = np.array([0, 270, 180, 90]) * np.pi / 180

        self.multi_test = False

    def load_state_dict(self, state_dict):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        super().load_state_dict(state_dict, strict=False)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, image, return_auxilary=False, **kwargs):
        _, _, H, W = image.shape
        c2, c3, c4, c5 = self.resnet_backbone(image)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)

        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)

        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)

        output = F.interpolate(output, size=(H, W), **self._up_kwargs)

        return output

    def init_weights(self, pretrained=""):
        if os.path.isfile(pretrained):
            state_dict = torch.load(pretrained)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
            print("==> Load pretrained weight", pretrained)
