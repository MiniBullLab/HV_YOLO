import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.shufflenetv2 import ShuffleNetV2
from base_model.mobilenetv2 import MobileNetV2
from base_model.resnet import ResNet

def main():
    print("process start...")
    model = ResNet()
    model.printBlockName()
    print("process end!")

if __name__ == '__main__':
    main()