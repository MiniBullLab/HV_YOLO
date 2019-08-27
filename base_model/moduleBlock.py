import torch.nn.functional as F
import numpy as np
from .baseLayer import GlobalAvgPool2d
from .baseBlock import *

class ConvBNReLU(BaseBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
                              stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU()
        self.setModelName("ConvBNReLU")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvBNLeakyReLU(BaseBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,\
                              stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.setModelName("ConvBNLeakyReLU")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyRelu(x)
        return x

# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
class InvertedResidual(BaseBlock):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)
        self.setModelName("InvertedResidual")

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# -----------------------------------------------------------------
#                      For Resnet
# -----------------------------------------------------------------
class residualBasciNeck(BaseBlock):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.setModelName("residualBasciNeck")

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class residualBottleneck(BaseBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(residualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.setModelName("residualBottleneck")

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

        out += residual
        out = self.relu(out)

        return out

# -----------------------------------------------------------------
#                      New addition SELayer
# -----------------------------------------------------------------
class SEBlock(BaseBlock):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())
        self.setModelName("SEBlock")

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# check the module
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()

        self.conv1 = residualBottleneck(16, 4, 1)
        # self.conv1 = residualBasciNeck(16, 16, 1)
        # self.conv1 = InvertedResidual(16, 16, 1, 2)
        self.globalPool = GlobalAvgPool2d()
        self.se = SEBlock(16)
        # self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, input):

        x = self.conv1(input)

        return x

if __name__ == "__main__":
    import netron
    import torch.onnx as onnx

    input = torch.randn(1, 16, 32, 32)
    model = testNet()
    out = model(input)
    onnx.export(model, input, "./a.onnx")
    netron.start("./a.onnx", port=9999, host='localhost')
