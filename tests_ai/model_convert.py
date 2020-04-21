import os
import sys
import torch
sys.path.insert(0, os.getcwd() + "/..")

from easyai.model.utility.model_factory import ModelFactory

def main(cfgPath, modelPath, savePath):
    model_factory = ModelFactory()
    model = model_factory.get_model("../cfg/seg/fgsegv2.cfg")
    modelCvt = torch.load("/home/wfw/HASCO/all_wights/vgg16_FgSegNetV2.pt")
    if isinstance(modelCvt, dict):
        modelCvt = modelCvt['model']

    vs = []
    for i, (k, v) in enumerate(modelCvt.items()):
        vs.append(v)

    for j, (k_, v_) in enumerate(model.state_dict().items()):
        if j < 21:
            print(j, k_, vs[j].shape, v_.shape)
            v_.copy_(vs[j])

    # checkpoint = {'epoch': None,
    #               'best_value': None,
    #               'model': model.state_dict(),
    #               'optimizer': None}
    # torch.save(checkpoint, savePath)

    print("End of game!!!")


if __name__ == '__main__':
    #options = parse_arguments()
    #main(options.cfg)
    main("./cfg/yolov3-spp-dilation_BerkeleyAll.cfg", "./snapshot/detect/pretain.pt", "/home/wfw/HASCO/all_wights/vgg16_FgSegNetV2_2.pt")