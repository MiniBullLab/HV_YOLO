import os
import time
import cv2
from optparse import OptionParser
from utility.evaluatingOfmAp import *
from data_loader import *
from model.modelResultProcess import ModelResultProcess
from utility.torchModelProcess import TorchModelProcess
from utility.nonMaximumSuppression import *
from config import detectConfig

def parse_arguments():

    parser = OptionParser()
    parser.description = "This program test model"

    parser.add_option("-i", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")

    parser.add_option("-w", "--weights", dest="weights",
                      metavar="PATH", type="string", default="weights/latest.pt",
                      help="path to store weights")

    (options, args) = parser.parse_args()

    if options.valPath:
        if not os.path.exists(options.valPath):
            parser.error("Could not find the input val file")
        else:
            options.input_path = os.path.normpath(options.valPath)
    else:
        parser.error("'valPath' option is required to run this program")

    return options

def detectTest(valPath, cfgPath, weights_path):
    os.system('rm -rf ' + 'results')
    os.makedirs('results', exist_ok=True)

    trainDataProcess = TrainDataProcess()
    torchModelProcess = TorchModelProcess()
    modelResultProcess = ModelResultProcess()

    model = torchModelProcess.initModel(cfgPath, 0)
    model.setFreezeBn(True)

    evaluator = MeanApEvaluating(valPath, detectConfig.className)

    torchModelProcess.loadLatestModelWeight(weights_path, model)
    torchModelProcess.modelTestInit(model)

    dataloader = ImageDetectValDataLoader(valPath, batch_size=1, img_size=detectConfig.imgSize)

    prev_time = time.time()
    for i, (img_path, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')

        # Get detections
        with torch.no_grad():
            output = model(img.to(torchModelProcess.getDevice()))
            pred = modelResultProcess.detectResult(model, output, 5e-3)
            # print(pred)
            detections = non_max_suppression(pred, 5e-3, detectConfig.nmsThresh)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

        path, fileNameAndPost = os.path.split(img_path)
        fileName, post = os.path.splitext(fileNameAndPost)
        img = cv2.imread(img_path)

        detectObjects = trainDataProcess.resizeDetectObjects(img, detectConfig.imgSize, detections,
                                             detectConfig.className)

        for object in detectObjects:
            confidence = object.classConfidence * object.objectConfidence
            x1 = object.min_corner.x
            y1 = object.min_corner.y
            x2 = object.max_corner.x
            y2 = object.max_corner.y
            with open("./results/comp4_det_test_" + object.name + ".txt", 'a') as file:
                file.write(
                    "{} {} {} {} {} {}\n".format(fileName, confidence, x1, y1, x2, y2))

    mAP, aps = evaluator.do_python_eval("./results/", "./results/comp4_det_test_")

    return mAP, aps

def main():
    print("process start...")
    torch.cuda.empty_cache()
    options = parse_arguments()
    detectTest(options.valPath, options.cfg, options.weights)
    print("process end!")

if __name__ == '__main__':
    main()