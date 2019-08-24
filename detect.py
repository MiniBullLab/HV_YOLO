import argparse
import time
from data_loader import *
from model.modelParse import ModelParse
from utility.datasets import *
from utility.utils import *
from utility.torchModelProcess import TorchModelProcess
from utility.nonMaximumSuppression import *
from drawing.colorDefine import ColorDefine
import config.config as config

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

parser = argparse.ArgumentParser()
# Get data configuration

parser.add_argument('-image_folder', type=str, default='data/samples', help='path to images')
parser.add_argument('-weight_name', type=str, default='yolov3.pt', help='path to your model')

parser.add_argument('-cfg', type=str, default=f_path + 'cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-conf_thres', type=float, default=0.50, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
opt = parser.parse_args()
print(opt)


def main(opt):
    opt.img_size = [640, 352]
    torchModelProcess = TorchModelProcess()
    # Load model
    modelParse = ModelParse()
    model = modelParse.parse(opt.cfg, freezeBn=True)

    weights_path = f_path + 'weights/' + opt.weight_name
    print("Loading checkpoint {}".format(weights_path))
    if torch.cuda.device_count() > 1:
        checkpoint = torchModelProcess.convert_state_dict(torch.load(weights_path,
                                                                     map_location='cpu')['model'])
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    del checkpoint

    model.to(device).eval()

    dataloader = ImagesLoader(opt.image_folder, batch_size=1, img_size=opt.img_size)
    #dataloader = VideoLoader(opt.image_folder, batch_size=1, img_size=opt.img_size)

    prev_time = time.time()
    for i, (oriImg, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')

        # Get detections
        detections = []
        with torch.no_grad():
            output = model(img.to(device))
            preds = []
            for i in range(0, 3):
                predEach = model.lossList[i](output[i])
                preds.append(predEach)
            pred = torch.cat(preds, 1)
            pred = pred[pred[:, :, 4] > opt.conf_thres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), opt.conf_thres, opt.nms_thres)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

        # The amount of padding that was added
        pad_x = 0 if (opt.img_size[0]/oriImg.shape[1]) < (opt.img_size[1]/oriImg.shape[0]) else opt.img_size[0] - opt.img_size[1] / oriImg.shape[0] * oriImg.shape[1]
        pad_y = 0 if (opt.img_size[0]/oriImg.shape[1]) > (opt.img_size[1]/oriImg.shape[0]) else opt.img_size[1] - opt.img_size[0] / oriImg.shape[1] * oriImg.shape[0]

        # Image height and width after padding is removed
        unpad_h = opt.img_size[1] - pad_y
        unpad_w = opt.img_size[0] - pad_x

        # Draw bounding boxes and labels of detections
        if detections:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(ColorDefine.colors, len(unique_classes))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
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


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
