# ShuffleNetV2 in PyTorch.
# See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
from .baseModelName import BaseModelName
from .baseModel import *
from base_block.blockName import BatchNormType, ActivationType, BlockType, LossType
from base_block.utilityBlock import ConvBNActivationBlock, ConvActivationBlock

__all__ = ['shufflenet_v2_1_0']

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C/g), H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        xChannel = x.size(1)
        if torch.is_tensor(xChannel):
            xChannel = np.asarray(xChannel)

        c = int(xChannel * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)

        self.convBnReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn2 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          groups=in_channels,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.convBnReLU3 = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.convBnReLU1(x2)
        out = self.convBn2(out)
        out = self.convBnReLU3(out)
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2,
                 bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.convBn1l = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=in_channels,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.convBnReLU2l = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=mid_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)
        # right
        self.convBnReLU1r = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=mid_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.convBn2r = ConvBNActivationBlock(in_channels=mid_channels,
                                          out_channels=mid_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=mid_channels,
                                          bnName=bnName,
                                          activationName=ActivationType.Linear)

        self.convBnReLU3r = ConvBNActivationBlock(in_channels=mid_channels,
                                          out_channels=mid_channels,
                                          kernel_size=1,
                                          bnName=bnName,
                                          activationName=activationName)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        outl = self.convBnReLU2l(self.convBn1l(x))
        # right
        outr = self.convBnReLU3r(self.convBn2r(self.convBnReLU1r(x)))
        # concat
        out = torch.cat([outl, outr], 1)
        out = self.shuffle(out)
        return out

class ShuffleNetV2(BaseModel):
    def __init__(self, data_channel=3, num_blocks=[3,7,3], out_channels=(24, 24, 48, 96, 192), stride=[2,2,2],
                 dilation=[1,1,1], bnName=BatchNormType.BatchNormalize, activationName=ActivationType.ReLU):
        super().__init__()
        # init param
        self.setModelName(BaseModelName.ShuffleNetV2)
        self.data_channel = data_channel
        self.num_blocks = num_blocks # [3, 7, 3]
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.activationName = activationName
        self.bnName = bnName

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
        self.layer3 = self._make_layer(self.out_channels[2], self.num_blocks[0], self.stride[0], self.dilation[0], bnName, activationName)
        self.layer4 = self._make_layer(self.out_channels[3], self.num_blocks[1], self.stride[0], self.dilation[1], bnName, activationName)
        self.layer5 = self._make_layer(self.out_channels[4], self.num_blocks[2], self.stride[0], self.dilation[2], bnName, activationName)

    def _make_layer(self, out_channels, num_blocks, stride, dilation, bnName, activationName):
        layers = [DownBlock(self.in_channels, out_channels, stride=stride,
                 bnName=bnName, activationName=activationName)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dilation=dilation,
                 bnName=bnName, activationName=activationName))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []
        out = self.layer1(x)
        blocks.append(out)
        out = self.layer2(out)
        blocks.append(out)
        out = self.layer3(out)
        blocks.append(out)
        out = self.layer4(out)
        blocks.append(out)
        out = self.layer5(out)
        blocks.append(out)

        return blocks

def get_shufflenet_v2(pretrained=False, **kwargs):
    model = ShuffleNetV2()

    if pretrained:
        raise ValueError("Not support pretrained")
    return model

def shufflenet_v2_1_0(**kwargs):
    return get_shufflenet_v2(**kwargs)
