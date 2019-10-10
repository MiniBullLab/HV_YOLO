import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from model.model_factory import ModelFactory

def main():
    model_factory = ModelFactory()
    model = model_factory.getModelFromCfg('../cfg/shufflenetV2-0.5_spp_BerkeleyAll.cfg')
    print(model)

if __name__ == '__main__':
    main()