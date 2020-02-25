import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.seg.segment import Segmentation


def main():
    segment = Segmentation("../cfg/seg/fgsegv2.cfg", 0)
    segment.load_weights("/home/wfw/EDGE/HV_YOLO/tests_ai/log/snapshot/model_epoch_197.pt") # "./log/snapshot/model_epoch_8.pt"
    segment.process("/home/wfw/data/VOCdevkit/LED_segment/JPEGImages")

if __name__ == '__main__':
    main()