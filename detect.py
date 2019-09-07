import os
import time
from optparse import OptionParser
from data_loader import *
from utility.utils import *
from model.modelResultProcess import ModelResultProcess
from utility.torchModelProcess import TorchModelProcess
from drawing.imageDraw import ImageDraw
from utility.nonMaximumSuppression import *
from config import detectConfig

def parse_arguments():

    parser = OptionParser()
    parser.description = "This program detect"

    parser.add_option("-i", "--input", dest="inputPath",
                      metavar="PATH", type="string", default=None,
                      help="images path or video path")

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")

    parser.add_option("-w", "--weights", dest="weights",
                      metavar="PATH", type="string", default="weights/latest.pt",
                      help="path to store weights")

    (options, args) = parser.parse_args()

    if options.inputPath:
        if not os.path.exists(options.inputPath):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.inputPath)
    else:
        parser.error("'input' option is required to run this program")

    return options

def detect(imageFolder, cfgPath, weightsPath):


    imageDraw = ImageDraw()
    trainDataProcess = TrainDataProcess()
    torchModelProcess = TorchModelProcess()
    modelResultProcess = ModelResultProcess()

    model = torchModelProcess.initModel(cfgPath, 0)
    model.setFreezeBn(True)

    torchModelProcess.loadLatestModelWeight(weightsPath, model)
    torchModelProcess.modelTestInit(model)

    dataloader = ImagesLoader(imageFolder, batch_size=1, img_size=detectConfig.imgSize)
    #dataloader = VideoLoader(opt.image_folder, batch_size=1, img_size=opt.img_size)

    for i, (oriImg, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')
        prev_time = time.time()
        # Get detections
        with torch.no_grad():
            output = model(img.to(torchModelProcess.getDevice()))
            pred = modelResultProcess.detectResult(model, output, detectConfig.confThresh)
            #print(pred)
            detections = non_max_suppression(pred, detectConfig.confThresh, detectConfig.nmsThresh)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))

        detectObjects = trainDataProcess.resizeDetectObjects(oriImg, detectConfig.imgSize, detections,
                                                             detectConfig.className)
        imageDraw.drawDetectObjects(oriImg, detectObjects)

        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(oriImg.shape[1] * 0.8), int(oriImg.shape[0] * 0.8))
        cv2.imshow("image", oriImg)
        cv2.waitKey()

def main():
    print("process start...")
    torch.cuda.empty_cache()
    options = parse_arguments()
    detect(options.inputPath, options.cfg, options.weights)
    print("process end!")


if __name__ == '__main__':
    main()
