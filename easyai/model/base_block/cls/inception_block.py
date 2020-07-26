#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class InceptionBlockName():

    InceptionStem = "inceptionStem"
    InceptionA = "inceptionA"
    ReductionA = "reductionA"
    InceptionB = "inceptionB"
    ReductionB = "reductionB"
    InceptionC = "inceptionC"
    InceptionResNetA = "inceptionResNetA"
    InceptionResNetB = "inceptionResNetB"
    InceptionResNetC = "inceptionResNetC"
    InceptionResNetReductionA = "inceptionResNetReductionA"
    InceptionResNetReductionB = "inceptionResNetReductionB"


class InceptionStem(BaseBlock):

    # """Figure 3. The schema for stem of the pure Inception-v4 and
    # Inception-ResNet-v2 networks. This is the input part of those
    # networks."""
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionStem)
        self.conv1 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 32, kernel_size=3,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(32, 32, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(32, 64, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3_conv = ConvBNActivationBlock(64, 96, kernel_size=3, padding=1,
                                                    bnName=bn_name, activationName=activation_name)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)

        self.branch7x7a = nn.Sequential(
            ConvBNActivationBlock(160, 64, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(64, 64, kernel_size=(7, 1), padding=(3, 0),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(64, 64, kernel_size=(1, 7), padding=(0, 3),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(64, 96, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch7x7b = nn.Sequential(
            ConvBNActivationBlock(160, 64, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(64, 96, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = ConvBNActivationBlock(192, 192, kernel_size=3,
                                                 stride=1, padding=1,
                                                 bnName=bn_name,
                                                 activationName=activation_name)

    def forward(self, x):
        x = self.conv1(x)

        x = [
            self.branch3x3_conv(x),
            self.branch3x3_pool(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branch7x7a(x),
            self.branch7x7b(x)
        ]
        x = torch.cat(x, 1)

        x = [
            self.branchpoola(x),
            self.branchpoolb(x)
        ]

        x = torch.cat(x, 1)

        return x


class InceptionA(BaseBlock):

    # """Figure 4. The schema for 35 × 35 grid modules of the pure
    # Inception-v4 network. This is the Inception-A block of Figure 9."""
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionA)

        self.branch3x3stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, 64, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(64, 96, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(96, 96, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 64, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(64, 96, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch1x1 = ConvBNActivationBlock(input_channel, 96, kernel_size=1,
                                               bnName=bn_name, activationName=activation_name)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNActivationBlock(input_channel, 96, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name)
        )

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class ReductionA(BaseBlock):

    # """Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
    # Different variants of this blocks (with various number of filters)
    # are used in Figure 9, and 15 in each of the new Inception(-v4, - ResNet-v1,
    # -ResNet-v2) variants presented in this paper. The k, l, m, n numbers
    # represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channel, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.ReductionA)
        self.branch3x3stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, out_channels[0], kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(out_channels[0], out_channels[1], kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(out_channels[1], out_channels[2], kernel_size=3, stride=2,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3 = ConvBNActivationBlock(input_channel, out_channels[3], kernel_size=3, stride=2,
                                               bnName=bn_name, activationName=activation_name)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channel = input_channel + out_channels[2] + out_channels[3]

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionB(BaseBlock):

    # """Figure 5. The schema for 17 × 17 grid modules of the pure Inception-v4 network.
    # This is the Inception-B block of Figure 9."""
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionB)

        self.branch7x7stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, 192, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(192, 192, kernel_size=(1, 7), padding=(0, 3),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(192, 224, kernel_size=(7, 1), padding=(3, 0),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(224, 224, kernel_size=(1, 7), padding=(0, 3),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(224, 256, kernel_size=(7, 1), padding=(3, 0),
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch7x7 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 192, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(192, 224, kernel_size=(1, 7), padding=(0, 3),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(224, 256, kernel_size=(7, 1), padding=(3, 0),
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch1x1 = ConvBNActivationBlock(input_channel, 384, kernel_size=1,
                                               bnName=bn_name, activationName=activation_name)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNActivationBlock(input_channel, 128, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class ReductionB(BaseBlock):

    # """Figure 8. The schema for 17 × 17 to 8 × 8 grid-reduction mod- ule.
    # This is the reduction module used by the pure Inception-v4 network in
    # Figure 9."""
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.ReductionB)
        self.branch7x7 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 256, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(256, 256, kernel_size=(1, 7), padding=(0, 3),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(256, 320, kernel_size=(7, 1), padding=(3, 0),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(320, 320, kernel_size=3, stride=2, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 192, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(192, 192, kernel_size=3, stride=2, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionC(BaseBlock):

    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        # """Figure 6. The schema for 8×8 grid modules of the pure
        # Inceptionv4 network. This is the Inception-C block of Figure 9."""

        super().__init__(InceptionBlockName.InceptionC)

        self.branch3x3stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, 384, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(384, 448, kernel_size=(1, 3), padding=(0, 1),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(448, 512, kernel_size=(3, 1), padding=(1, 0),
                                  bnName=bn_name, activationName=activation_name),
        )
        self.branch3x3stacka = ConvBNActivationBlock(512, 256, kernel_size=(1, 3), padding=(0, 1),
                                                     bnName=bn_name, activationName=activation_name)
        self.branch3x3stackb = ConvBNActivationBlock(512, 256, kernel_size=(3, 1), padding=(1, 0),
                                                     bnName=bn_name, activationName=activation_name)

        self.branch3x3 = ConvBNActivationBlock(input_channel, 384, kernel_size=1,
                                               bnName=bn_name, activationName=activation_name)
        self.branch3x3a = ConvBNActivationBlock(384, 256, kernel_size=(3, 1), padding=(1, 0),
                                                bnName=bn_name, activationName=activation_name)
        self.branch3x3b = ConvBNActivationBlock(384, 256, kernel_size=(1, 3), padding=(0, 1),
                                                bnName=bn_name, activationName=activation_name)

        self.branch1x1 = ConvBNActivationBlock(input_channel, 256, kernel_size=1,
                                               bnName=bn_name, activationName=activation_name)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNActivationBlock(input_channel, 256, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name)
        )

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)


class InceptionResNetA(BaseBlock):

    # """Figure 16. The schema for 35 × 35 grid (Inception-ResNet-A)
    # module of the Inception-ResNet-v2 network."""
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionResNetA)
        self.branch3x3stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, 32, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(32, 48, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(48, 64, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 32, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(32, 32, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch1x1 = ConvBNActivationBlock(input_channel, 32, kernel_size=1,
                                               bnName=bn_name,
                                               activationName=activation_name)

        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channel, 384, kernel_size=1)
        self.bn = NormalizeLayer(bn_name=bn_name, out_channel=384)
        self.relu = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch3x3stack(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)

        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output


class InceptionResNetB(BaseBlock):

    # """Figure 17. The schema for 17 × 17 grid (Inception-ResNet-B) module of
    # the Inception-ResNet-v2 network."""
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionResNetB)
        self.branch7x7 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 128, kernel_size=1,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(128, 160, kernel_size=(1, 7), padding=(0, 3),
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(160, 192, kernel_size=(7, 1), padding=(3, 0),
                                  bnName=bn_name,
                                  activationName=activation_name)
        )

        self.branch1x1 = ConvBNActivationBlock(input_channel, 192, kernel_size=1,
                                               bnName=bn_name, activationName=activation_name)

        self.reduction1x1 = nn.Conv2d(384, 1154, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channel, 1154, kernel_size=1)

        self.bn = NormalizeLayer(bn_name=bn_name, out_channel=1154)
        self.relu = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch7x7(x)
        ]

        residual = torch.cat(residual, 1)

        # """In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals
        # before their being added to the accumulated layer activations (cf. Figure 20)."""
        residual = self.reduction1x1(residual) * 0.1

        shortcut = self.shortcut(x)

        output = self.bn(residual + shortcut)
        output = self.relu(output)

        return output


class InceptionResNetC(BaseBlock):

    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        # Figure 19. The schema for 8×8 grid (Inception-ResNet-C)
        # module of the Inception-ResNet-v2 network."""
        super().__init__(InceptionBlockName.InceptionResNetC)
        self.branch3x3 = nn.Sequential(
            ConvBNActivationBlock(input_channel, 192, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(192, 224, kernel_size=(1, 3), padding=(0, 1),
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(224, 256, kernel_size=(3, 1), padding=(1, 0),
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch1x1 = ConvBNActivationBlock(input_channel, 192, kernel_size=1,
                                               bnName=bn_name, activationName=activation_name)
        self.reduction1x1 = nn.Conv2d(448, 2048, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channel, 2048, kernel_size=1)

        self.bn = NormalizeLayer(bn_name=bn_name, out_channel=2048)
        self.relu = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch3x3(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1

        shorcut = self.shorcut(x)

        output = self.bn(shorcut + residual)
        output = self.relu(output)

        return output


class InceptionResNetReductionA(BaseBlock):

    # """Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
    # Different variants of this blocks (with various number of filters)
    # are used in Figure 9, and 15 in each of the new Inception(-v4, - ResNet-v1,
    # -ResNet-v2) variants presented in this paper. The k, l, m, n numbers
    # represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channel, out_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionResNetReductionA)
        self.branch3x3stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, out_channels[0], kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(out_channels[0], out_channels[1], kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(out_channels[1], out_channels[2], kernel_size=3, stride=2,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3 = ConvBNActivationBlock(input_channel, out_channels[3], kernel_size=3, stride=2,
                                               bnName=bn_name, activationName=activation_name)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channel = input_channel + out_channels[2] + out_channels[3]

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionResNetReductionB(BaseBlock):

    # """Figure 18. The schema for 17 × 17 to 8 × 8 grid-reduction module.
    # Reduction-B module used by the wider Inception-ResNet-v1 network in
    # Figure 15."""
    # I believe it was a typo(Inception-ResNet-v1 should be Inception-ResNet-v2)
    def __init__(self, input_channel,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(InceptionBlockName.InceptionResNetReductionB)
        self.branchpool = nn.MaxPool2d(3, stride=2)

        self.branch3x3a = nn.Sequential(
            ConvBNActivationBlock(input_channel, 256, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(256, 384, kernel_size=3, stride=2,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3b = nn.Sequential(
            ConvBNActivationBlock(input_channel, 256, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(256, 288, kernel_size=3, stride=2,
                                  bnName=bn_name, activationName=activation_name)
        )

        self.branch3x3stack = nn.Sequential(
            ConvBNActivationBlock(input_channel, 256, kernel_size=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(256, 288, kernel_size=3, padding=1,
                                  bnName=bn_name, activationName=activation_name),
            ConvBNActivationBlock(288, 320, kernel_size=3, stride=2,
                                  bnName=bn_name, activationName=activation_name)
        )

    def forward(self, x):
        x = [
            self.branch3x3a(x),
            self.branch3x3b(x),
            self.branch3x3stack(x),
            self.branchpool(x)
        ]

        x = torch.cat(x, 1)
        return x
