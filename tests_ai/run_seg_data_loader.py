import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import cv2
import numpy as np
from easyai.data_loader.seg.segment_dataloader import get_segment_train_dataloader
from easyai.config import segment_config
from sklearn.utils import compute_class_weight

def decode_segmap(temp):

    colors = [#[  0,   0,   0],
              [128,  64, 128],
              [244,  35, 232],
              [ 70,  70,  70],
              [102, 102, 156],
              [190, 153, 153],
              [153, 153, 153],
              [250, 170,  30],
              [220, 220,   0],
              [107, 142,  35],
              [152, 251, 152],
              [  0, 130, 180],
              [220,  20,  60],
              [255,   0,   0],
              [  0,   0, 142],
              [  0,   0,  70],
              [  0,  60, 100],
              [  0,  80, 100],
              [  0,   0, 230],
              [119,  11,  32]]

    label_colours = dict(zip(range(19), colors))

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))

    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb

def main():
    # Get dataloader
    dataloader = get_segment_train_dataloader("/home/wfw/data/VOCdevkit/LED/ImageSets/train.txt", segment_config.imgSize,
                                              segment_config.train_batch_size)

    for i, (imgs, segs) in enumerate(dataloader):
        for img, seg in zip(imgs, segs):
            print("Image: {}".format(i))
            seg_num = seg.numpy().flatten()
            print(seg_num[seg_num==0].size, seg_num[seg_num==1].size)

            lb = np.unique(seg_num)  # 0., 1
            cls_weight = compute_class_weight('balanced', lb, seg_num)
            class_0 = cls_weight[0]
            class_1 = cls_weight[1] if len(lb) > 1 else 1.0
            print(class_0, class_1)

            img = img.numpy()
            img = np.transpose(img, (1, 2, 0)).copy()
            seg = decode_segmap(seg.numpy())
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            cv2.imshow('seg', seg)
            key = cv2.waitKey()
            if key == 27:
                break

if __name__ == '__main__':
    main()