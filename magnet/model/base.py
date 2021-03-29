import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
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

class RefinementBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(RefinementBottleneck, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False),
            BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False),
            BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        )
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
        #                        bias=False)
        # self.bn3 = BatchNorm2d(planes * self.expansion,
        #                        momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        # residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        import pdb; pdb.set_trace()
        if self.downsample is not None:
            return self.relu(self.conv(x) + self.downsample(x))
        return self.relu(self.conv(x) + x)

        # if self.downsample is not None:
        #     out =  out + self.downsample(x)

        # out = self.relu(out)

        # return out