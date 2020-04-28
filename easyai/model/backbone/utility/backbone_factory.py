#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os.path
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.cls.mobilenetv2 import mobilenetv2_1_0
from easyai.model.backbone.cls.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from easyai.model.backbone.cls.mixnet import mixnet_s, mixnet_m, mixnet_l
from easyai.model.backbone.cls.shufflenetv2 import shufflenetv2_1_0
from easyai.model.backbone.cls.resnet import resnet18, resnet34
from easyai.model.backbone.cls.resnet import resnet50, resnet101, resnet152
from easyai.model.backbone.cls.darknet import darknet21, darknet53
from easyai.model.backbone.cls.darknet import darknet21_dilated8, darknet21_dilated16
from easyai.model.backbone.cls.darknet import darknet53_dilated8, darknet53_dilated16
from easyai.model.backbone.cls.googlenet import GoogleNet
from easyai.model.backbone.cls.inceptionv4 import inceptionv4, inception_resnetv2
from easyai.model.backbone.cls.vgg import vgg11, vgg13, vgg16, vgg19
from easyai.model.backbone.cls.squeezenet import SqueezeNet, DilatedSqueezeNet
from easyai.model.backbone.cls.densenet import densenet121, densenet201, densenet169, densenet161
from easyai.model.backbone.cls.densenet import densenet121_dilated8, densenet121_dilated16
from easyai.model.backbone.cls.senet import se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152
from easyai.model.backbone.cls.sknet import sknet50, sknet101
from easyai.model.backbone.cls.resnext import resnext50, resnext101, resnext152
from easyai.model.backbone.cls.nasnet import nasnet
from easyai.model.backbone.cls.preact_resnet import preactresnet18, preactresnet34, preactresnet50
from easyai.model.backbone.cls.preact_resnet import preactresnet101, preactresnet152
from easyai.model.backbone.cls.xception import xception
from easyai.model.backbone.cls.xception1 import Xception65, XceptionA
from easyai.model.backbone.cls.attention_net import attention_net56, attention_net92
from easyai.model.backbone.cls.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2
from easyai.model.backbone.cls.efficientnet import efficientnet_b3, efficientnet_b4, efficientnet_b5
from easyai.model.backbone.cls.efficientnet import efficientnet_b6, efficientnet_b7
from easyai.model.backbone.cls.dpn import dpn26, dpn92
from easyai.model.backbone.cls.dfnet import dfnetv1, dfnetv2
from easyai.model.backbone.cls.pnasnet import pnasnet_A, pnasnet_B
from easyai.model.backbone.cls.wider_resnet import wider_resnet16, wider_resnet20, wider_resnet38
from easyai.model.backbone.cls.wider_resnet import wider_resnet16_a2, wider_resnet20_a2, wider_resnet38_a2
from easyai.model.backbone.cls.ghostnet import ghost_net
from easyai.model.backbone.cls.hrnet import hrnet_w18_small, hrnet_w18_small_v2, hrnet_w18
from easyai.model.backbone.cls.hrnet import hrnet_w30, hrnet_w32, hrnet_w40, hrnet_w44
from easyai.model.backbone.cls.hrnet import hrnet_w48, hrnet_w64
from easyai.model.backbone.utility.my_backbone import MyBackbone
from easyai.model.utility.model_parse import ModelParse


class BackboneFactory():

    def __init__(self):
        self.cfgReader = ModelParse()

    def get_base_model(self, backbone_name, default_args=None):
        input_name = backbone_name.strip()
        if input_name.endswith("cfg"):
            result = self.get_backbone_from_cfg(input_name)
        else:
            result = self.get_backbone_from_name(input_name, default_args)
            if result is None:
                print("backbone:%s error" % input_name)
        return result

    def get_backbone_from_cfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.cfgReader.readCfgFile(cfg_path)
        model = MyBackbone(model_define)
        model.set_name(file_name)
        return model

    def get_backbone_from_name(self, net_name, default_args=None):
        result = None
        net_name = net_name.strip()
        data_channel = default_args['data_channel']
        if net_name == BackboneName.ShuffleNetV2_1_0:
            result = shufflenetv2_1_0(data_channel)
        elif net_name == BackboneName.MobileNetv3_small:
            result = mobilenetv3_small(data_channel)
        elif net_name == BackboneName.MobileNetv3_large:
            result = mobilenetv3_large(data_channel)
        elif net_name == BackboneName.mixnet_s:
            result = mixnet_s(data_channel)
        elif net_name == BackboneName.mixnet_m:
            result = mixnet_m(data_channel)
        elif net_name == BackboneName.mixnet_l:
            result = mixnet_l(data_channel)
        elif net_name == BackboneName.MobileNetV2_1_0:
            result = mobilenetv2_1_0(data_channel)
        elif net_name == BackboneName.ResNet18:
            result = resnet18(data_channel)
        elif net_name == BackboneName.ResNet34:
            result = resnet34(data_channel)
        elif net_name == BackboneName.ResNet50:
            result = resnet50(data_channel)
        elif net_name == BackboneName.ResNet101:
            result = resnet101(data_channel)
        elif net_name == BackboneName.ResNet152:
            result = resnet152(data_channel)
        elif net_name == BackboneName.Darknet53:
            result = darknet53(data_channel)
        elif net_name == BackboneName.Darknet21:
            result = darknet21(data_channel)
        elif net_name == BackboneName.Darknet21_Dilated8:
            result = darknet21_dilated8(data_channel)
        elif net_name == BackboneName.Darknet21_Dilated16:
            result = darknet21_dilated16(data_channel)
        elif net_name == BackboneName.Darknet53_Dilated8:
            result = darknet53_dilated8(data_channel)
        elif net_name == BackboneName.Darknet53_Dilated16:
            result = darknet53_dilated16(data_channel)
        elif net_name == BackboneName.GoogleNet:
            result = GoogleNet(data_channel)
        elif net_name == BackboneName.InceptionV4:
            result = inceptionv4(data_channel)
        elif net_name == BackboneName.InceptionResNetV2:
            result = inception_resnetv2(data_channel)
        elif net_name == BackboneName.Vgg11:
            result = vgg11(data_channel)
        elif net_name == BackboneName.Vgg13:
            result = vgg13(data_channel)
        elif net_name == BackboneName.Vgg16:
            result = vgg16(data_channel)
        elif net_name == BackboneName.Vgg19:
            result = vgg19(data_channel)
        elif net_name == BackboneName.SqueezeNet:
            result = SqueezeNet(data_channel)
        elif net_name == BackboneName.DilatedSqueezeNet:
            result = DilatedSqueezeNet(data_channel)
        elif net_name == BackboneName.Densenet121:
            result = densenet121(data_channel)
        elif net_name == BackboneName.Densenet121_Dilated8:
            result = densenet121_dilated8(data_channel)
        elif net_name == BackboneName.Densenet121_Dilated16:
            result = densenet121_dilated16(data_channel)
        elif net_name == BackboneName.Densenet169:
            result = densenet169(data_channel)
        elif net_name == BackboneName.Densenet201:
            result = densenet201(data_channel)
        elif net_name == BackboneName.Densenet161:
            result = densenet161(data_channel)
        elif net_name == BackboneName.SEResNet18:
            result = se_resnet18(data_channel)
        elif net_name == BackboneName.SEResNet34:
            result = se_resnet34(data_channel)
        elif net_name == BackboneName.SEResNet50:
            result = se_resnet50(data_channel)
        elif net_name == BackboneName.SEResNet101:
            result = se_resnet101(data_channel)
        elif net_name == BackboneName.SEResNet152:
            result = se_resnet152(data_channel)
        elif net_name == BackboneName.SKNet50:
            result = sknet50(data_channel)
        elif net_name == BackboneName.SKNet101:
            result = sknet101(data_channel)
        elif net_name == BackboneName.ResNext50:
            result = resnext50(data_channel)
        elif net_name == BackboneName.ResNext101:
            result = resnext101(data_channel)
        elif net_name == BackboneName.ResNext152:
            result = resnext152(data_channel)
        elif net_name == BackboneName.NasNet:
            result = nasnet(data_channel)
        elif net_name == BackboneName.PreActResNet18:
            result = preactresnet18(data_channel)
        elif net_name == BackboneName.PreActResNet34:
            result = preactresnet34(data_channel)
        elif net_name == BackboneName.PreActResNet50:
            result = preactresnet50(data_channel)
        elif net_name == BackboneName.PreActResNet101:
            result = preactresnet101(data_channel)
        elif net_name == BackboneName.PreActResNet152:
            result = preactresnet152(data_channel)
        elif net_name == BackboneName.Xception:
            result = xception(data_channel)
        elif net_name == BackboneName.Xception65:
            result = Xception65(data_channel)
        elif net_name == BackboneName.XceptionA:
            result = XceptionA(data_channel)
        elif net_name == BackboneName.AttentionNet56:
            result = attention_net56(data_channel)
        elif net_name == BackboneName.AttentionNet92:
            result = attention_net92(data_channel)
        elif net_name == BackboneName.Efficientnet_b0:
            result = efficientnet_b0(data_channel)
        elif net_name == BackboneName.Efficientnet_b1:
            result = efficientnet_b1(data_channel)
        elif net_name == BackboneName.Efficientnet_b2:
            result = efficientnet_b2(data_channel)
        elif net_name == BackboneName.Efficientnet_b3:
            result = efficientnet_b3(data_channel)
        elif net_name == BackboneName.Efficientnet_b4:
            result = efficientnet_b4(data_channel)
        elif net_name == BackboneName.Efficientnet_b5:
            result = efficientnet_b5(data_channel)
        elif net_name == BackboneName.Efficientnet_b6:
            result = efficientnet_b6(data_channel)
        elif net_name == BackboneName.Efficientnet_b7:
            result = efficientnet_b7(data_channel)
        elif net_name == BackboneName.DPN26:
            result = dpn26(data_channel)
        elif net_name == BackboneName.DPN92:
            result = dpn92(data_channel)
        elif net_name == BackboneName.DFNetV1:
            result = dfnetv1(data_channel)
        elif net_name == BackboneName.DFNetV2:
            result = dfnetv2(data_channel)
        elif net_name == BackboneName.PNASNetA:
            result = pnasnet_A(data_channel)
        elif net_name == BackboneName.PNASNetB:
            result = pnasnet_B(data_channel)
        elif net_name == BackboneName.wider_resnet16:
            result = wider_resnet16(data_channel)
        elif net_name == BackboneName.wider_resnet20:
            result = wider_resnet20(data_channel)
        elif net_name == BackboneName.wider_resnet38:
            result = wider_resnet38(data_channel)
        elif net_name == BackboneName.wider_resnet16_a2:
            result = wider_resnet16_a2(data_channel)
        elif net_name == BackboneName.wider_resnet20_a2:
            result = wider_resnet20_a2(data_channel)
        elif net_name == BackboneName.wider_resnet38_a2:
            result = wider_resnet38_a2(data_channel)
        elif net_name == BackboneName.GhostNet:
            result = ghost_net(data_channel)
        elif net_name == BackboneName.HRnet_w18_small:
            result = hrnet_w18_small(data_channel)
        elif net_name == BackboneName.HRnet_w18_small_v2:
            result = hrnet_w18_small_v2(data_channel)
        elif net_name == BackboneName.HRnet_w18:
            result = hrnet_w18(data_channel)
        elif net_name == BackboneName.HRnet_w30:
            result = hrnet_w30(data_channel)
        elif net_name == BackboneName.HRnet_w32:
            result = hrnet_w32(data_channel)
        elif net_name == BackboneName.HRnet_w40:
            result = hrnet_w40(data_channel)
        elif net_name == BackboneName.HRnet_w44:
            result = hrnet_w44(data_channel)
        elif net_name == BackboneName.HRnet_w48:
            result = hrnet_w48(data_channel)
        elif net_name == BackboneName.HRnet_w64:
            result = hrnet_w64(data_channel)
        return result

