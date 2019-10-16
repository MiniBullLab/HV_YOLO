from optparse import OptionParser
from base_model.baseModelFactory import BaseModelFactory
from model.model_factory import ModelFactory
from drawing.modelNetShow import ModelNetShow

def parseArguments():

    parser = OptionParser()
    parser.description = "This program show model net"

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")

def main(cfgPath):
    base_model_factory = BaseModelFactory()
    model_factory = ModelFactory()
    modelNetShow = ModelNetShow(3, 640, 352)
    #model = model_factory.getModelFromCfg(cfgPath)
    model = base_model_factory.get_base_model_from_name("ResNet")
    #model = model_factory.getModelFromName("MobileV2FCN")
    modelNetShow.showNet(model)

if __name__ == '__main__':
    #options = parseArguments()
    #main(options.cfg)
    main("./cfg/mobileFCN.cfg")