import time
import cv2
from optparse import OptionParser
from model.modelsShuffleNet import *
from utils.evaluatingOfmAp import *
from data_loader import *

import config.config as config

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def parse_arguments():

    parser = OptionParser()
    parser.description = "This program test model"

    parser.add_option("-b", "--batch_size'", dest="batch_size",
                      type="int", default=16,
                      help="size of each image batch")

    parser.add_option("-c", "--cfg", dest="cfg",
                      metavar="PATH", type="string", default="cfg/yolov3.cfg",
                      help="cfg file path")

    parser.add_option("-i", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    parser.add_option("-w", "--weights", dest="weights",
                      metavar="PATH", type="string", default="weights/latest.pt",
                      help="path to store weights")

    (options, args) = parser.parse_args()

    return options

def main(cfg, weights_path, img_size, imageFile):
    os.system('rm -rf ' + 'results')
    os.makedirs('results', exist_ok=True)
    # Load model
    # Darknet53
    # model = Darknet(opt.cfg, opt.img_size)
    # ShuffleNetV2_1.0
    model = ShuffleYolo(cfg, img_size)

    evaluator = MeanApEvaluating(imageFile)

    # YoloLoss
    yoloLoss = []
    for m in model.module_list:
        for layer in m:
            if isinstance(layer, YoloLoss):
                yoloLoss.append(layer)

    if torch.cuda.device_count() > 1:
        checkpoint = convert_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    del checkpoint

    model.to(device).eval()

    # Set Dataloader
    dataloader = ImageDetectValDataLoader(imageFile, batch_size=1, img_size=img_size)

    prev_time = time.time()
    for i, (img_path, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')

        # Get detections
        with torch.no_grad():
            output = model(torch.from_numpy(img).unsqueeze(0).to(device))
            preds = []
            for i in range(0, 3):
                predEach = yoloLoss[i](output[i])
                preds.append(predEach)
            pred = torch.cat(preds, 1)
            pred = pred[pred[:, :, 4] > 5e-3]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), 5e-3, config.iouThresh) # select nms method (or, and, soft-nms)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

        img = cv2.imread(img_path)
        # The amount of padding that was added
        pad_x = 0 if (img_size[0]/img.shape[1]) < (img_size[1]/img.shape[0]) else img_size[0] - img_size[1] / img.shape[0] * img.shape[1]
        pad_y = 0 if (img_size[0]/img.shape[1]) > (img_size[1]/img.shape[0]) else img_size[1] - img_size[0] / img.shape[1] * img.shape[0]
        # Image height and width after padding is removed
        unpad_h = img_size[1] - pad_y
        unpad_w = img_size[0] - pad_x

        path, fileNameAndPost = os.path.split(img_path)
        fileName, post = os.path.splitext(fileNameAndPost)

        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round().item()
                x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                x2 = (x1 + box_w).round().item()
                y2 = (y1 + box_h).round().item()
                x1, y1, x2, y2 = max(x1, 1.0), max(y1, 1.0), max(x2, 0), max(y2, 0)

                # write to file
                for i in range(0, len(config.className)):
                    if int(cls_pred.cpu().numpy()) == i:
                        with open("./results/comp4_det_test_" + config.className[i] + ".txt", 'a') as file:
                            file.write(
                                "{} {} {} {} {} {}\n".format(fileName, cls_conf * conf, x1, y1, x2, y2))

    mAP, aps = evaluator.do_python_eval("./results/", "./results/comp4_det_test_")

    return mAP, aps

if __name__ == '__main__':
    torch.cuda.empty_cache()
    options = parse_arguments()
    img_size = 608
    main(options.cfg, options.weights_path, img_size, options.valPath)
