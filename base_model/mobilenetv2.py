# MobileNetV2 in PyTorch.
from .baseModel import *
from .baseLayer import ConvBNReLU, InvertedResidual

class MobileNetV2(BaseModel):
    def __init__(self, width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.setModelName("MobileNetV2")
        layer1_setting = [
            # t, c, n, s
            [1, 16, 1, 1]]
        layer2_setting = [
            [6, 24, 2, 2]]
        layer3_setting = [
            [6, 32, 3, 2]]
        layer4_setting = [
            [6, 64, 4, 2],
            [6, 96, 3, 1]]
        layer5_setting = [
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        # building first layer
        self.out_channels = (16, 24, 32, 96, 320)
        self.in_channels = int(32 * width_mult) if width_mult > 1.0 else 32
        self.conv1 = ConvBNReLU(3, self.in_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)

        # building inverted residual blocks
        self.layer1 = self._make_layer(InvertedResidual, layer1_setting, width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(InvertedResidual, layer2_setting, width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(InvertedResidual, layer3_setting, width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(InvertedResidual, layer4_setting, width_mult,
                                           dilation=2, norm_layer=norm_layer)
            self.layer5 = self._make_layer(InvertedResidual, layer5_setting, width_mult,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(InvertedResidual, layer4_setting, width_mult, norm_layer=norm_layer)
            self.layer5 = self._make_layer(InvertedResidual, layer5_setting, width_mult, norm_layer=norm_layer)

        self._init_weight()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for t, c, n, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            layers.append(block(self.in_channels, out_channels, stride, t, dilation, norm_layer=norm_layer))
            self.in_channels = out_channels
            for i in range(n - 1):
                layers.append(block(self.in_channels, out_channels, 1, t, 1, norm_layer=norm_layer))
                self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []
        x = self.conv1(x)
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        x = self.layer5(x)
        blocks.append(x)
        return blocks

    def _init_weight(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_mobilenet_v2(width_mult=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = MobileNetV2(width_mult=width_mult, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def mobilenet_v2_1_0(**kwargs):
    return get_mobilenet_v2(1.0, **kwargs)


def mobilenet_v2_0_75(**kwargs):
    return get_mobilenet_v2(0.75, **kwargs)


def mobilenet_v2_0_5(**kwargs):
    return get_mobilenet_v2(0.5, **kwargs)


def mobilenet_v2_0_25(**kwargs):
    return get_mobilenet_v2(0.25, **kwargs)


if __name__ == '__main__':
    import torch
    model = MobileNetV2()
    dummy_input = torch.randn(10, 3, 224, 224)


