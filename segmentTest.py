from config import configSegment
import time
from data_loader import *
from utility.utils import *
from utility.torchModelProcess import TorchModelProcess
from loss.enetLoss import cross_entropy2dDet
from utility.metrics import runningScore

def main(cfgPath, weights_path, img_size, valPath):

    running_metrics = runningScore(2)
    torchModelProcess = TorchModelProcess()

    model = torchModelProcess.initModel(cfgPath, 0)
    model.setFreezeBn(True)
    torchModelProcess.loadLatestModelWeight(weights_path, model)
    torchModelProcess.modelTestInit(model)

    dataloader = ImageSegmentValDataLoader(valPath, batch_size=configSegment.test_batch_size, img_size=img_size)
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

    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    running_metrics.reset()

    return score, class_iou
