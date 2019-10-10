from .baseBlock import *
from .baseLayer import GlobalAvgPool2d
from .blockName import BlockType, ActivationType, BatchNormType
from .activationFunction import ActivationFunction
from .myBatchNorm import MyBatchNormalize

class ConvActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvActivationBlock)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=True)
        self.activation = ActivationFunction.getFunction(activationName)

    def forward(self, x):
        convX = self.conv(x)
        activationX = self.activation(convX)
        return activationX

class ConvBNActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=False)
        self.bn = MyBatchNormalize.getFunction(bnName, out_channels)
        self.activation = ActivationFunction.getFunction(activationName)

    def forward(self, x):
        convX = self.conv(x)
        bnX = self.bn(convX)
        activationX = self.activation(bnX)
        return activationX

# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
class InvertedResidual(BaseBlock):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 dilation=1, bnName=BatchNormType.BatchNormalize, **kwargs):
        super().__init__(BlockType.InvertedResidual)
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            convBNReLUBlock = ConvBNActivationBlock(in_channels, inter_channels, 1,\
                                                    activationName=ActivationType.ReLU6, bnName=bnName)
            layers.append(convBNReLUBlock)
        layers.extend([
            # dw
            ConvBNActivationBlock(inter_channels, inter_channels, 3, \
                                  stride, dilation, dilation, groups=inter_channels,\
                                  activationName=ActivationType.ReLU6, bnName=bnName),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            MyBatchNormalize.getFunction(bnName, out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# -----------------------------------------------------------------
#                      For Resnet
# -----------------------------------------------------------------
class ResidualBasciNeck(BaseBlock):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__(BlockType.ResidualBasciNeck)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

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

class ResidualBottleneck(BaseBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__(BlockType.ResidualBottleneck)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        out += residual
        out = self.relu(out)

        return out

# -----------------------------------------------------------------
#                      New addition SELayer
# -----------------------------------------------------------------
class SEBlock(BaseBlock):
    def __init__(self, channel, reduction=16):
        super().__init__(BlockType.SEBlock)
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

if __name__ == "__main__":
    pass
