import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import torch
from easyai.tasks.seg.segment_train import SegmentionTrain
from easyai.tools.model_net_show import ModelNetShow
from easyai.model.backbone.utility.backbone_factory import BackboneFactory


def main():
    show = ModelNetShow()

    # back_bone_factory = BackboneFactory()
    # back_bone = back_bone_factory.get_base_model_from_cfg("../cfg/seg/vgg16.cfg")
    # back_bone.print_block_name()
    # back_bone_list = back_bone.get_outchannel_list()

    # show.show_process.show_from_model(back_bone)

    input_x = torch.randn(1, 3, 352, 640)
    show.show_process.set_input(input_x)
    model = show.model_factory.get_model("../cfg/seg/fgsegv2.cfg")
    show.show_process.show_from_model(model)

if __name__ == '__main__':
    main()