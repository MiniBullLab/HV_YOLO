from .baseBlock import *
from .baseLayer import GlobalAvgPool2d
from .blockName import BlockType, ActivationType, BatchNormType
from .activationFunction import ActivationFunction
from .myBatchNorm import MyBatchNormalize

class ConvActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=True)
        activation = ActivationFunction.getFunction(activationName)
        self.block = nn.Sequential(OrderedDict([
            (BlockType.Convolutional, conv),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x

class ConvBNActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=False)
        bn = MyBatchNormalize.getFunction(bnName, out_channels)
        activation = ActivationFunction.getFunction(activationName)
        self.block = nn.Sequential(OrderedDict([
            (BlockType.Convolutional, conv),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x

class InvertedResidual(BaseBlock):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 dilation=1, bnName=BatchNormType.BatchNormalize, **kwargs):
        super().__init__(BlockType.InvertedResidual)
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = OrderedDict()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            conv1 = ConvBNActivationBlock(in_channels, inter_channels, 1,\
                                                    activationName=ActivationType.ReLU6,
                                                    bnName=bnName)
            conv2 = ConvBNActivationBlock(inter_channels, inter_channels, 3, \
                                  stride, dilation, dilation, groups=inter_channels, \
                                  activationName=ActivationType.ReLU6, bnName=bnName)
            conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
            bn = MyBatchNormalize.getFunction(bnName, out_channels)
            layer_name = "%s_1" % BlockType.ConvBNActivationBlock
            layers[layer_name] = conv1
            layer_name = "%s_2" % BlockType.ConvBNActivationBlock
            layers[layer_name] = conv2
            layer_name = BlockType.Convolutional
            layers[layer_name] = conv3
            layer_name = bnName
            layers[layer_name] = bn

        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class SEBlock(BaseBlock):
    def __init__(self, channel, reduction=16):
        super().__init__(BlockType.SEBlock)
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

if __name__ == "__main__":
    pass
