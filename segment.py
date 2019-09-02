import time, cv2, os
from optparse import OptionParser
from utility.utils import *
import scipy.misc as misc
from data_loader import *
from utility.torchModelProcess import TorchModelProcess
from drawing.imageDraw import ImageDraw

from config import configSegment

def parse_arguments():

    parser = OptionParser()
    parser.description = "This program detect"

    parser.add_option("-i", "--input", dest="inputPath",
                      metavar="PATH", type="string", default=None,
                      help="images path or video path")

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/mobileFCN.cfg",
                      help="cfg file path")

    parser.add_option("-w", "--weights", dest="weights",
                      metavar="PATH", type="string", default="snapshot/latest.pt",
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

def segment(imageFolder, cfgPath, weightsPath):
    torchModelProcess = TorchModelProcess()
    model = torchModelProcess.initModel(cfgPath, 0)
    model.setFreezeBn(True)
    torchModelProcess.loadLatestModelWeight(weightsPath, model)
    torchModelProcess.modelTestInit(model)

    #dataloader = ImagesLoader(imageFolder, batch_size=1, img_size=configSegment.imgSize)
    dataloader = VideoLoader(imageFolder, batch_size=1, img_size=configSegment.imgSize)

    prev_time = time.time()
    for i, (oriImg, img) in enumerate(dataloader):
        # Get detections
        with torch.no_grad():
            output = model(img.to(torchModelProcess.getDevice()))[0]

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

        # oriImage shape, input shape
        ori_w, ori_h = oriImg.shape[1], oriImg.shape[0]
        pre_h, pre_w = configSegment.imgSize[1], configSegment.imgSize[0]

        # segmentation
        segmentations = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
        ratio = min(float(pre_w) / ori_w, float(pre_h) / ori_h)  # ratio  = old / new
        new_shape = [round(ori_h * ratio), round(ori_w * ratio)]
        dw = pre_w - new_shape[1]  # width padding
        dh = pre_h - new_shape[0]  # height padding
        padH = [dh // 2, dh - (dh // 2)]
        padW = [dw // 2, dw - (dw // 2)]
        segmentations = segmentations[padH[0]:pre_h - padH[1], padW[0]:pre_w - padW[1]]
        segmentations = segmentations.astype(np.float32)
        segmentations = misc.imresize(segmentations, [ori_h, ori_w], 'nearest',
                                      mode='F')  # float32 with F mode, resize back to orig_size
        drawSeg = ImageDraw()
        decoded = drawSeg.drawSegmentResult(oriImg, segmentations, configSegment.className)

        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(decoded.shape[1] * 0.5), int(decoded.shape[0] * 0.5))
        cv2.imshow('image', decoded)

        if cv2.waitKey() & 0xFF == 27:
            break

def main():
    print("process start...")
    torch.cuda.empty_cache()
    options = parse_arguments()
    segment(options.inputPath, options.cfg, options.weights)
    print("process end!")

if __name__ == '__main__':
    main()
