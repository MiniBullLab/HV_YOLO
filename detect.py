import os
import time
from optparse import OptionParser
from data_loader import *
from utility.utils import *
from utility.torchModelProcess import TorchModelProcess
from utility.torchDeviceProcess import TorchDeviceProcess
from utility.nonMaximumSuppression import *
import config.config as config

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
    torchModelProcess = TorchModelProcess()

    model = torchModelProcess.initModel(cfgPath)
    model.setFreezeBn(True)

    torchModelProcess.loadLatestModelWeight(weightsPath, model)
    torchModelProcess.modelTestInit(model)

    dataloader = ImagesLoader(imageFolder, batch_size=1, img_size=config.imgSize)
    #dataloader = VideoLoader(opt.image_folder, batch_size=1, img_size=opt.img_size)

    for i, (oriImg, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')
        prev_time = time.time()
        # Get detections
        detections = []
        with torch.no_grad():
            output = model(img.to(TorchDeviceProcess.device))
            preds = []
            for i in range(0, 3):
                predEach = model.lossList[i](output[i])
                preds.append(predEach)
            pred = torch.cat(preds, 1)
            pred = pred[pred[:, :, 4] > config.confThresh]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), config.confThresh, config.nmsThresh)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))

        # The amount of padding that was added
        pad_x = 0 if (config.imgSize[0]/oriImg.shape[1]) < (config.imgSize[1]/oriImg.shape[0]) else config.imgSize[0] - config.imgSize[1] / oriImg.shape[0] * oriImg.shape[1]
        pad_y = 0 if (config.imgSize[0]/oriImg.shape[1]) > (config.imgSize[1]/oriImg.shape[0]) else config.imgSize[1] - config.imgSize[0] / oriImg.shape[1] * oriImg.shape[0]

        # Image height and width after padding is removed
        unpad_h = config.imgSize[1] - pad_y
        unpad_w = config.imgSize[0] - pad_x

        # Draw bounding boxes and labels of detections
        if detections[0] is not None:
            # unique_classes = detections[:, -1].cpu().unique()
            # bbox_colors = random.sample(ColorDefine.colors, len(unique_classes))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * oriImg.shape[0]
                box_w = ((x2 - x1) / unpad_w) * oriImg.shape[1]
                y1 = (((y1 - pad_y // 2) / unpad_h) * oriImg.shape[0]).round().item()
                x1 = (((x1 - pad_x // 2) / unpad_w) * oriImg.shape[1]).round().item()
                x2 = (x1 + box_w).round().item()
                y2 = (y1 + box_h).round().item()
                x1, y1, x2, y2 = max(x1, 1.0), max(y1, 1.0), min(x2, oriImg.shape[1]-1.0), min(y2, oriImg.shape[0]-1.0)
                cv2.rectangle(oriImg, (int(x1), int(y1)), (int(x2), int(y2)), [0,255,0], 2)

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
