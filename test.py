import argparse
import time, cv2

from models import *
from modelsShuffleNet import *
from utils.utils import *
from data_loader import *

import config.config as config

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def main(opt):
    os.system('rm -rf ' + 'results')
    os.makedirs('results', exist_ok=True)
    # Load model
    # Darknet53
    # model = Darknet(opt.cfg, opt.img_size)
    # ShuffleNetV2_1.0
    model = ShuffleYolo(opt.cfg, opt.img_size)

    # YoloLoss
    yoloLoss = []
    for m in model.module_list:
        for layer in m:
            if isinstance(layer, YoloLoss):
                yoloLoss.append(layer)

    if torch.cuda.device_count() > 1:
        checkpoint = convert_state_dict(torch.load(opt.weights_path, map_location='cpu')['model'])
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(opt.weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    del checkpoint

    model.to(device).eval()

    # Set Dataloader
    dataloader = ImageDetectValDataLoader(opt.image_folder, batch_size=1, img_size=opt.img_size)

    prev_time = time.time()
    for i, (img_paths, img) in enumerate(dataloader):
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

        img = cv2.imread(img_paths[0])
        # The amount of padding that was added
        pad_x = 0 if (opt.img_size[0]/img.shape[1]) < (opt.img_size[1]/img.shape[0]) else opt.img_size[0] - opt.img_size[1] / img.shape[0] * img.shape[1]
        pad_y = 0 if (opt.img_size[0]/img.shape[1]) > (opt.img_size[1]/img.shape[0]) else opt.img_size[1] - opt.img_size[0] / img.shape[1] * img.shape[0]
        # Image height and width after padding is removed
        unpad_h = opt.img_size[1] - pad_y
        unpad_w = opt.img_size[0] - pad_x

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
                                "{} {} {} {} {} {}\n".format(img_paths[0].split("/")[-1][:-4], cls_conf * conf, x1, y1, x2, y2))

    # check file exists
    for i in range(0, len(config.className)):
        if not os.path.exists("./results/comp4_det_test_" + config.className[i] + ".txt"):
            lostfile = open("./results/comp4_det_test_" + config.className[i] + ".txt", 'w')
            lostfile.write(
                "{} {} {} {} {} {}\n".format(img_paths[0].split("/")[-1][:-4], 0, 0, 0, 0, 0))
            lostfile.close()

    mAP, aps = do_python_eval("./results/", "./data/imagelist.txt", "./results/comp4_det_test_", "/home/wfw/data/VOCdevkit/BerkeleyDet/Annotations/")

    return mAP, aps

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
