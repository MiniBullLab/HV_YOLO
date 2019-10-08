'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from .baseModelName import BaseModelName
from .baseModel import *
from base_block.blockName import BatchNormType, ActivationType, BlockType, LossType
from base_block.utilityBlock import ConvBNActivationBlock, ConvActivationBlock
from base_block.activationFunction import ActivationFunction

__all__ = ['resnet_18']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super(BasicBlock, self).__init__()

        self.convBnReLU1 =  ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=planes,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn2 = ConvBNActivationBlock(in_channels=planes,
                                          out_channels=planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=dilation,
                                          dilation=dilation,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*planes:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.ReLU = ActivationFunction.getFunction(activationName)

    def forward(self, x):
        out = self.convBnReLU1(x)
        out = self.convBn2(out)
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super(Bottleneck, self).__init__()

        self.convBnReLU1 =  ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=planes,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBnReLU2 = ConvBNActivationBlock(in_channels=planes,
                                          out_channels=planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=dilation,
                                          dilation=dilation,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn3 =  ConvBNActivationBlock(in_channels=planes,
                                          out_channels=self.expansion*planes,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*planes:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=self.expansion*planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.ReLU = ActivationFunction.getFunction(activationName)

    def forward(self, x):
        out = self.convBnReLU1(x)
        out = self.convBnReLU2(out)
        out = self.convBn3(out)
        out += self.shortcut(x)
        out = self.ReLU(out)
        return out

class ResNet(nn.Module):
    def __init__(self, data_channel=3, num_blocks=[2,2,2,2], out_channels=[64,64,128,256,512], stride=[1,2,2,2],
                 dilation=[1,1,1,1], bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU, block=BasicBlock, **kwargss):
        super().__init__()
        # init param
        # self.setModelName(BaseModelName.MobileNetV2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks # [3, 7, 3]
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activationName = activationName
        self.bnName = bnName
        self.block = block

        # init layer
        self.layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                          out_channels=self.out_channels[0],
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          bnName=self.bnName,
                                          activationName=self.activationName)


        self.layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_channels = self.out_channels[1]
        self.layer3 = self._make_layer(self.out_channels[1], self.num_blocks[0], self.stride[0], self.dilation[0], bnName, activationName, block)
        self.layer4 = self._make_layer(self.out_channels[2], self.num_blocks[1], self.stride[1], self.dilation[1], bnName, activationName, block)
        self.layer5 = self._make_layer(self.out_channels[3], self.num_blocks[2], self.stride[2], self.dilation[2], bnName, activationName, block)
        self.layer6 = self._make_layer(self.out_channels[4], self.num_blocks[3], self.stride[3], self.dilation[3], bnName, activationName, block)

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