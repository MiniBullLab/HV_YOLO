'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from .baseModelName import BaseModelName
from .baseModel import *

__all__ = ['resnet_18']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, data_channel=3, num_blocks=[2,2,2,2], out_channels=[64,64,128,256,512], stride=[1,2,2,2],
                 dilation=[1,1,1,1], BatchNorm=nn.BatchNorm2d, activation=nn.ReLU, groups=2, block=BasicBlock, **kwargss):
        super().__init__()
        # init param
        # self.setModelName(BaseModelName.MobileNetV2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks # [3, 7, 3]
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.BatchNorm = BatchNorm
        self.groups = groups
        self.block = block

        # init layer
        self.layer1 = nn.Sequential(nn.Conv2d(self.data_channel, self.out_channels[0], kernel_size=3,
                                    stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(self.out_channels[0]),
                                    nn.ReLU())
        self.layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = self.out_channels[0]
        self.layer3 = self._make_layer(self.out_channels[1], self.num_blocks[0], self.stride[0], self.dilation[0], BatchNorm, activation, block)
        self.layer4 = self._make_layer(self.out_channels[2], self.num_blocks[1], self.stride[1], self.dilation[1], BatchNorm, activation, block)
        self.layer5 = self._make_layer(self.out_channels[3], self.num_blocks[2], self.stride[2], self.dilation[2], BatchNorm, activation, block)
        self.layer6 = self._make_layer(self.out_channels[4], self.num_blocks[3], self.stride[3], self.dilation[3], BatchNorm, activation, block)

    def _make_layer(self, out_channels, num_blocks, stride, dilation, BatchNorm, activation, block):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels, 1))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []
        out = self.layer1(x)
        blocks.append(out)
        out = self.layer2(out)
        blocks.append(out)
        out = self.layer3(out)
        out = self.layer4(out)
        blocks.append(out)
        out = self.layer5(out)
        blocks.append(out)
        out = self.layer6(out)
        blocks.append(out)

        return blocks

def get_resnet(pretrained=False, root='~/.torch/models', **kwargs):
    model = ResNet()

    if pretrained:
        raise ValueError("Not support pretrained")
    return model

def resnet_18(**kwargs):
    return get_resnet(**kwargs)