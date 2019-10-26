from optparse import OptionParser
from base_model.base_model_factory import BaseModelFactory
from model.model_factory import ModelFactory
from drawing.modelNetShow import ModelNetShow


def parse_arguments():

    parser = OptionParser()
    parser.description = "This program show model net"

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")


def main(cfgPath):
    base_model_factory = BaseModelFactory()
    model_factory = ModelFactory()
    modelNetShow = ModelNetShow(3, 640, 352)
    model = model_factory.get_model(cfgPath)
    #model = base_model_factory.get_base_model_from_cfg("./cfg/darknet-tiny.cfg")
    #model = base_model_factory.get_base_model_from_name("MobileNetV2")
    #model = model_factory.get_model("MobileV2FCN")
    #model = model_factory.get_model("./cfg/mobileFCN.cfg")
    modelNetShow.showNet(model)

if __name__ == '__main__':
    #options = parse_arguments()
    #main(options.cfg)
    main("./cfg/cifar100.cfg")
