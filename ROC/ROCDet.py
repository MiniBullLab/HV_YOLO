import config
import argparse
import time
from utils.datasets import *
from utils.utils import *
from models import *
from modelsShuffleNet import *
from loss import *

import matplotlib.pyplot as plt

def drawROC(output, labelNames):
    for labelName in labelNames:
        f=open(output+labelName+'.txt', 'r')
        results=f.readlines()

        acc = []
        rec = []
        for result in results:
            obj = result[:-1].split(">>")
            acc.append(float(obj[1].split(":")[1]))
            rec.append(float(obj[2].split(":")[1]))

        plt.plot(acc, rec, 'r*-')#, color = color, linewidth = linewidth)
        plt.plot([0.0,1.1], [0.0,1.1], 'b-')
        plt.xlim(0.0,1.1)
        plt.ylim(0.0,1.1)
        plt.title('ROC for '+ labelName +' In YOLOV3')
        plt.xlabel('ACC')
        plt.ylabel('REC')
        plt.show()

def prCal(image_folder, weights_path, outputPath, annotation_path, thresh):
    os.system('rm -rf ' + outputPath)
    os.makedirs(outputPath, exist_ok=True)
    # init model
    # model = ShuffleMTSD(config.net_config_path, config.imgSize)
    model = ShuffleYolo(config.net_config_path, config.imgSize)

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

    model.cuda().eval()

    # Set Dataloader
    dataloader = load_images(image_folder, batch_size=1, img_size=config.imgSize)

    prev_time = time.time()
    for i, (img_paths, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')

        # Get detections
        with torch.no_grad():
            output = model(torch.from_numpy(img).unsqueeze(0).cuda())
            preds = []
            for i in range(0, 3):
                predEach = yoloLoss[i](output[i])
                preds.append(predEach)
            pred = torch.cat(preds, 1)
            pred = pred[pred[:, :, 4] > thresh]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), thresh, config.iouThresh) # select nms method (or, and, soft-nms)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

        img = cv2.imread(img_paths[0])
        # The amount of padding that was added
        pad_x = 0 if (config.imgSize[0]/img.shape[1]) < (config.imgSize[1]/img.shape[0]) else config.imgSize[0] - config.imgSize[1] / img.shape[0] * img.shape[1]
        pad_y = 0 if (config.imgSize[0]/img.shape[1]) > (config.imgSize[1]/img.shape[0]) else config.imgSize[1] - config.imgSize[0] / img.shape[1] * img.shape[0]
        # Image height and width after padding is removed
        unpad_h = config.imgSize[1] - pad_y
        unpad_w = config.imgSize[0] - pad_x

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
                        with open(outputPath + "/comp4_det_test_" + config.className[i] + ".txt", 'a') as file:
                            file.write(
                                "{} {} {} {} {} {}\n".format(img_paths[0].split("/")[-1][:-4], cls_conf * conf, x1, y1, x2, y2))

    # check file exists
    for i in range(0, len(config.className)):
        if not os.path.exists(outputPath + "/comp4_det_test_" + config.className[i] + ".txt"):
            lostfile = open(outputPath + "/comp4_det_test_" + config.className[i] + ".txt", 'w')
            lostfile.write(
                "{} {} {} {} {} {}\n".format(img_paths[0].split("/")[-1][:-4], 0, 0, 0, 0, 0))
            lostfile.close()

    rps = {}
    for i, cls in enumerate(config.className):
        if cls == '__background__':
            continue
        filename = outputPath + "/comp4_det_test_" + cls + '.txt'
        _, _, _, rec, prec = preRec(filename, annotation_path, "./data/imagelist.txt", cls, ovthresh=0.5, use_07_metric=False)
        rps[cls] = [rec, prec]

    return rps

def main(weights_path, image_folder, annotation_path, step):
    outputPath = './ROC/results/'
    rocResultPath = './ROC/rocResult/'
    os.system('rm -rf ' + rocResultPath)
    os.makedirs(rocResultPath, exist_ok=True)
    for thresh_conf in np.arange(0.01, 1, step):
        print("Testing with the thresh {:.2f}".format(thresh_conf))
        rps = prCal(weights_path=weights_path, image_folder=image_folder, outputPath=outputPath, annotation_path=annotation_path, thresh=thresh_conf)
        for cls in config.className:
            f = open(rocResultPath + cls + '.txt', 'a')
            f.write('Thresh: {:.2f} >> Accuracy: {:.3f} >> Recall: {:.3f} \n'.format(thresh_conf,
                rps[cls][1], rps[cls][0]))
            f.close()
    drawROC(rocResultPath, config.className)
    print("process end!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-path', type=str, help='path to test weights')
    parser.add_argument('--image-folder', type=str, help='path to test image')
    parser.add_argument('--annotation-path', type=str, help='path to image annotation')
    parser.add_argument('--step', type=float, default=0.02, help='thresh step')
    opt = parser.parse_args()

    main(opt.weights_path, opt.image_folder, opt.annotation_path, opt.step)
