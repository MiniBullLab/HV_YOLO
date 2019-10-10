import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.shufflenetv2 import ShuffleNetV2

def main():
    print("process start...")
    model = ShuffleNetV2()
    model.printBlockName()
    print("process end!")

if __name__ == '__main__':
    main()