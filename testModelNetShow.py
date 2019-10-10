from optparse import OptionParser
from model.model_factory import ModelFactory
from drawing.modelNetShow import ModelNetShow

def parseArguments():

    parser = OptionParser()
    parser.description = "This program show model net"

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")

def main(cfgPath):
    model_factory = ModelFactory()
    modelNetShow = ModelNetShow(3, 640, 352)
    model = model_factory.getModelFromCfg(cfgPath)
    #model = model_factory.getModelFromName("MobileV2FCN")
    modelNetShow.showNet(model)

if __name__ == '__main__':
    #options = parseArguments()
    #main(options.cfg)
    main("./cfg/mobileFCN.cfg")