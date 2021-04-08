import torch
import torch.nn as nn

from .base import BN_MOMENTUM, BatchNorm2d, Bottleneck


class RefinementMagNet(nn.Module):
    """Refinement module of MagNet

    Args:
        n_classes (int): no. classes
        use_bn (bool): use batch normalization on the input

    """

    def __init__(self, n_classes, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.bn0 = BatchNorm2d(n_classes * 2, momentum=BN_MOMENTUM)
        # 2 conv layers
        self.conv1 = nn.Conv2d(n_classes * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # 2 residual blocks
        self.residual = self._make_layer(Bottleneck, 64, 32, 2)

        # Prediction head
        self.seg_conv = nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make residual block"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, fine_segmentation, coarse_segmentation):
        x = torch.cat([fine_segmentation, coarse_segmentation], dim=1)
        if self.use_bn:
            x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.residual(x)

        return self.seg_conv(x)
