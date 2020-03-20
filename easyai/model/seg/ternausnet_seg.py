#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""BiSeNet
    Reference:
        V. Iglovikov and A. Shvets.
        "TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation".
        ArXiv e-prints:1801.05746 (2018).
"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.utility.cross_entropy2d import CrossEntropy2d
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.base_block.seg.ternausnet_block import DecoderBlock, DecoderBlockLinkNet
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
import torchvision


class UNet11(BaseModel):
    """
    :param num_classes:
    :param num_filters:
    :param pretrained:
        False - no pre-trained network used
        True - encoder pre-trained with VGG11
    """
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        super().__init__()
        self.set_name(ModelName.UNet11)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.pretrained = pretrained
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg11(pretrained=self.pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )

        self.center = DecoderBlock(256 + self.num_filters * 8, self.num_filters * 8 * 2,
                                   self.num_filters * 8, is_deconv=True)
        self.dec5 = DecoderBlock(512 + self.num_filters * 8, self.num_filters * 8 * 2,
                                 self.num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512 + self.num_filters * 8, self.num_filters * 8 * 2,
                                 self.num_filters * 4, is_deconv=True)
        self.dec3 = DecoderBlock(256 + self.num_filters * 4, self.num_filters * 4 * 2,
                                 self.num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128 + self.num_filters * 2, self.num_filters * 2 * 2,
                                 self.num_filters, is_deconv=True)
        self.dec1 = ConvActivationBlock(in_channels=64 + self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        padding=1,
                                        activationName=self.activation_name)

        self.final = nn.Conv2d(self.num_filters, self.num_classes, kernel_size=1)

    def create_loss(self, input_dict=None):
        pass

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class UNet16(BaseModel):
    """
    :param num_classes:
    :param num_filters:
    :param pretrained:
        False - no pre-trained network used
        True - encoder pre-trained with VGG16
    """
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        super().__init__()
        self.set_name(ModelName.UNet16)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.pretrained = pretrained
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=self.pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, self.num_filters * 8 * 2, self.num_filters * 8, is_deconv=False)

        self.dec5 = DecoderBlock(512 + self.num_filters * 8, self.num_filters * 8 * 2,
                                 self.num_filters * 8, is_deconv=False)
        self.dec4 = DecoderBlock(512 + self.num_filters * 8, self.num_filters * 8 * 2,
                                 self.num_filters * 8, is_deconv=False)
        self.dec3 = DecoderBlock(256 + self.num_filters * 8, self.num_filters * 4 * 2,
                                 self.num_filters * 2, is_deconv=False)
        self.dec2 = DecoderBlock(128 + self.num_filters * 2, self.num_filters * 2 * 2,
                                 self.num_filters, is_deconv=False)
        self.dec1 = ConvActivationBlock(in_channels=64 + self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        padding=1,
                                        activationName=self.activation_name)
        self.final = nn.Conv2d(self.num_filters, self.num_classes, kernel_size=1)

    def create_loss(self, input_dict=None):
        pass

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class AlbuNet(BaseModel):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
    """
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        super().__init__()
        self.set_name(ModelName.AlbuNet)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.pretrained = pretrained
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=self.pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, self.num_filters * 8 * 2, self.num_filters * 8, is_deconv=False)

        self.dec5 = DecoderBlock(512 + self.num_filters * 8, self.num_filters * 8 * 2,
                                 self.num_filters * 8, is_deconv=False)
        self.dec4 = DecoderBlock(256 + self.num_filters * 8, self.num_filters * 8 * 2,
                                 self.num_filters * 8, is_deconv=False)
        self.dec3 = DecoderBlock(128 + self.num_filters * 8, self.num_filters * 4 * 2,
                                 self.num_filters * 2, is_deconv=False)
        self.dec2 = DecoderBlock(64 + self.num_filters * 2, self.num_filters * 2 * 2,
                                 self.num_filters * 2 * 2, is_deconv=False)
        self.dec1 = DecoderBlock(self.num_filters * 2 * 2, self.num_filters * 2 * 2,
                                 self.num_filters, is_deconv=False)
        self.dec0 = ConvActivationBlock(in_channels=self.num_filters,
                                        out_channels=self.num_filters,
                                        kernel_size=3,
                                        padding=1,
                                        activationName=self.activation_name)
        self.final = nn.Conv2d(self.num_filters, self.num_classes, kernel_size=1)

    def create_loss(self, input_dict=None):
        pass

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out


class LinkNet34(BaseModel):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.set_name(ModelName.LinkNet34)
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.filters = [64, 128, 256, 512]
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        resnet = torchvision.models.resnet34(pretrained=self.pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(self.filters[3], self.filters[2])
        self.decoder3 = DecoderBlockLinkNet(self.filters[2], self.filters[1])
        self.decoder2 = DecoderBlockLinkNet(self.filters[1], self.filters[0])
        self.decoder1 = DecoderBlockLinkNet(self.filters[0], self.filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(self.filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, self.num_classes, 2, padding=1)

    def create_loss(self, input_dict=None):
        pass

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out