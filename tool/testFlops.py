import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from utility.model_summary import summary
from model.modelParse import ModelParse

def main():
    print("process start...")
    modelParse = ModelParse()
    model = modelParse.getModel("MobileV2FCN")
    summary(model, [1, 3, 640, 352])
    print("process end!")

if __name__ == '__main__':
    main()