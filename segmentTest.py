import os, time
from optparse import OptionParser
from data_loader import *
from utility.utils import *
from torch_utility.torchModelProcess import TorchModelProcess
from utility.metrics import runningScore
from config import segmentConfig

def parse_arguments():

    parser = OptionParser()
    parser.description = "This program test model"

    parser.add_option("-i", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/mobileFCN.cfg",
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

def segmentTest(valPath, cfgPath, weights_path):

    running_metrics = runningScore(2)
    torchModelProcess = TorchModelProcess()

    model = torchModelProcess.initModel(cfgPath, 0)
    model.setFreezeBn(True)
    torchModelProcess.loadLatestModelWeight(weights_path, model)
    torchModelProcess.modelTestInit(model)

    dataloader = ImageSegmentValDataLoader(valPath, batch_size=segmentConfig.test_batch_size,
                                           img_size=segmentConfig.imgSize)
    print("Eval data num: {}".format(len(dataloader)))

    prev_time = time.time()
    for i, (img, segMap_val) in enumerate(dataloader):
        # Get detections
        with torch.no_grad():
            output = model(img.cuda())[0]

            #------------seg---------------------
            pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            gt = segMap_val[0].data.cpu().numpy()
            running_metrics.update(gt, pred)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    running_metrics.reset()

    return score, class_iou

def main():
    print("process start...")
    torch.cuda.empty_cache()
    options = parse_arguments()
    segmentTest(options.valPath, options.cfg, options.weights)
    print("process end!")

if __name__ == '__main__':
    main()