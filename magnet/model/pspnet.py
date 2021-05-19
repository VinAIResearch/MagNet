import torch
from torch import nn
from torch.nn import functional as F

from .resnet import resnet101


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(input=stage(feats), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode="bilinear", align_corners=False)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=2048, pretrained=True):
        super().__init__()
        self.feats = resnet101(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        self.classifier = nn.Conv2d(deep_features_size, n_classes, kernel_size=1)

    def load_state_dict(self, state_dict):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        super().load_state_dict(state_dict, strict=False)

    def forward(self, x, return_auxilary=False):
        f = self.feats(x)[-1]
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = self.classifier(f)
        auxiliary = F.interpolate(auxiliary, size=x.size()[2:], mode="bilinear", align_corners=False)

        p = self.final(p)
        p = F.interpolate(p, size=x.size()[2:], mode="bilinear", align_corners=False)

        if return_auxilary:
            return p, auxiliary

        return (p + 0.4 * auxiliary) / 1.4
