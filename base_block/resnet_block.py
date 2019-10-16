from .baseBlock import *
from base_block.activationFunction import ActivationFunction
from base_block.blockName import BatchNormType, ActivationType
from base_block.utilityBlock import ConvBNActivationBlock

class ResnetNetBlockName():

    BasicBlock = "basicBlock"
    Bottleneck = "bottleneck"

class BasicBlock(BaseBlock):
    expansion = 1

    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__(ResnetNetBlockName.BasicBlock)

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

class Bottleneck(BaseBlock):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super(Bottleneck, self).__init__(ResnetNetBlockName.Bottleneck)

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