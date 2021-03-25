import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from scipy import ndimage

__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def rotate(x, degree):
    B, C, _, _ = x.shape
    outs = []
    for i in range(C):
        outs += [torch.from_numpy(ndimage.rotate(x[0, i], degree, reshape=False))]
    outs = torch.stack(outs)
    return outs.unsqueeze(0)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)
        
        x = self.maxpool(c1)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def custom_resnet(in_channels, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels)
    if pretrained:
        state = model.state_dict()
        partial = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "conv1" not in k}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    return model
        

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

class ResnetFPN(nn.Module):
    def __init__(self, n_labels, in_channels=3):
        super(ResnetFPN, self).__init__()
        self.resnet_backbone = resnet50(in_channels)
        self._up_kwargs = {'mode': 'bilinear'}
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
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
        self.classify = nn.Conv2d(128*4, n_labels, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
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
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, image, return_feat=False):
        _, _, H, W = image.shape
        c2, c3, c4, c5 = self.resnet_backbone(image)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        ps0 = [p5, p4, p3, p2]
        
        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)
        ps1 = [p5, p4, p3, p2]
        
        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)
        ps2 = [p5, p4, p3, p2]

        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)
        
        output = F.interpolate(output, size=(H, W), **self._up_kwargs)
        
        if return_feat:
            return output, p2
        return output
    
    def run_rotation(self, x, return_feat=False):
        pre_rotate = [0, 90, 180, 270]
        post_rotate = [0, 270, 180, 90]

        x = x.cpu()
        out = 0
        feat = 0
        for a,b in zip(pre_rotate, post_rotate):
            inp = rotate(x, a)
            temp_out, temp_feat = self.forward(inp.cuda(), return_feat=True)
            out = out + rotate(temp_out.cpu(), b).cuda()
            feat = feat + rotate(temp_feat.cpu(), b).cuda()
        if return_feat:
            return out, feat
        return out
