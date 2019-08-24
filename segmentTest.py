import config
import time
from data_loader import *
from utility.utils import *
from model.modelParse import ModelParse
from loss.enetLoss import cross_entropy2dDet

from utility.metrics import runningScore

def evalModel(evaluate_weight):

    running_metrics = runningScore(2)
    # init model
    modelParse = ModelParse()
    model = modelParse.parse("./cfg/mobileFCN.cfg")

    model.load_state_dict(evaluate_weight)
    model.cuda().eval()

    dataloader = ImageSegmentValDataLoader("/home/wfw/data/VOCdevkit/Tusimple/ImageSets/val.txt", batch_size=config.test_batch_size, img_size=[640, 352])
    print("Eval data num: {}".format(len(dataloader)))

    prev_time = time.time()
    for i, (img, segMap_val) in enumerate(dataloader):
        # print('%g/%g' % (i + 1, len(dataloader)), end=' ')

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
