from data_loader import *
import cv2
import numpy as np
import torch
from utility.utilss import *

from model.mobileV2FCN import *

# def decode_segmap(temp):
#     colors = [  # [  0,   0,   0],
#         [128, 64, 128],
#         [244, 35, 232],
#         [70, 70, 70],
#         [102, 102, 156],
#         [190, 153, 153],
#         [153, 153, 153],
#         [250, 170, 30],
#         [220, 220, 0],
#         [107, 142, 35],
#         [152, 251, 152],
#         [0, 130, 180],
#         [220, 20, 60],
#         [255, 0, 0],
#         [0, 0, 142],
#         [0, 0, 70],
#         [0, 60, 100],
#         [0, 80, 100],
#         [0, 0, 230],
#         [119, 11, 32]]
#
#     label_colours = dict(zip(range(19), colors))
#
#     r = temp.copy()
#     g = temp.copy()
#     b = temp.copy()
#     for l in range(0, 19):
#         r[temp == l] = label_colours[l][0]
#         g[temp == l] = label_colours[l][1]
#         b[temp == l] = label_colours[l][2]
#
#     rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
#
#     rgb[:, :, 0] = r / 255.0
#     rgb[:, :, 1] = g / 255.0
#     rgb[:, :, 2] = b / 255.0
#
#     return rgb
#
# def test():
#     # Get dataloader
#     dataloader = ImageSegmentTrainDataLoader("/home/wfw/data/VOCdevkit/Tusimple/ImageSets/train.txt", batch_size=1,
#                                          img_size=[640, 352], augment=True)
#     for i, (imgs, segs) in enumerate(dataloader):
#         for img, seg in zip(imgs, segs):
#             print("Image: {}".format(i))
#             img = img.numpy()
#             img = np.transpose(img, (1, 2, 0)).copy()
#             seg = decode_segmap(seg.numpy())
#             cv2.imshow('img', img)
#             key = cv2.waitKey(1)
#             cv2.imshow('seg', seg)
#             key = cv2.waitKey()
#             if key == 27:
#                 break
#
# if __name__=="__main__":
#     test()

model = MobileFCN("./cfg/mobileFCN.cfg", img_size=[640, 352])
# print(model)
#
dummy_input = torch.randn(1, 3, 640, 352)
y = model(dummy_input)
# torch.onnx.export(model, dummy_input, "./mobilFCN.onnx")
hyperparams = {}
hyperparams['channels']=3
hyperparams['height']=352
hyperparams['width']=640
summary(model, hyperparams)

# checkpoint = torch.load("/home/wfw/HASCO/HV_YOLO/weights/model_best.pth", map_location='cpu')['state_dict']
# model.load_state_dict(checkpoint)
# print(model.keys())
# model_state = model['state_dict']
# for k,v in model_state.items():
#     print(k, v.shape)
