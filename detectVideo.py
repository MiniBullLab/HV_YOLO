import argparse
import time

from models import *
from modelsShuffleNet import *
from utils.datasets import *
from utils.utils import *

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

parser = argparse.ArgumentParser()
# Get data configuration

parser.add_argument('-video_folder', type=str, default='data/samples', help='path to images')
parser.add_argument('-img_size', type=list, default=[640, 352], help='size of the image')
parser.add_argument('-weight_name', type=str, default='yolov3.pt', help='path to your model')
parser.add_argument('-output_folder', type=str, default='output', help='path to outputs')

parser.add_argument('-cfg', type=str, default=f_path + 'cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-class_path', type=str, default=f_path + 'data/berkeley.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.50, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
opt = parser.parse_args()
print(opt)

def main(opt):
    os.system('rm -rf ' + opt.output_folder)
    os.makedirs(opt.output_folder, exist_ok=True)

    # Load model
    model = ShuffleYolo(opt.cfg, opt.img_size)

    # YoloLoss
    yoloLoss = []
    for m in model.module_list:
        for layer in m:
            if isinstance(layer, YoloLoss):
                yoloLoss.append(layer)

    weights_path = f_path + 'weights/' + opt.weight_name
    if weights_path.endswith('.weights'):  # saved in darknet format
        load_weights(model, weights_path)
    else:  # endswith('.pt'), saved in pytorch format
        if weights_path.endswith('weights/yolov3.pt') and not os.path.isfile(weights_path):
            os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights_path)
            
        print("Loading checkpoint {}".format(weights_path))
        if torch.cuda.device_count() == 1:
            checkpoint = convert_state_dict(torch.load(weights_path, map_location='cpu')['model'])
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        del checkpoint

    model.to(device).eval()

    # Set Dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
    dataloader = load_video(opt.video_folder, batch_size=opt.batch_size, img_size=opt.img_size)

    prev_time = time.time()
    for i, (frames, img) in enumerate(dataloader):

        start_time = time.time()
        # Get detections
        with torch.no_grad():
            output = model(img.to(device))
            end_time = time.time()
            print("Using time {}".format(end_time - start_time))

            preds = []
            for i in range(0, 3):
                predEach = yoloLoss[i](output[i])
                preds.append(predEach)
            pred = torch.cat(preds, 1)

            # for i in range(0, opt.batch_size):
            #     pred = pred[0, :, :]
            #     pred = pred[pred[:, 4] > opt.conf_thres]
            #
            #     if len(pred) > 0:

            detections = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

            # print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            prev_time = time.time()

            for i in range(0, len(detections)):
                frame = frames[i]
                det = detections[i]

                if det is not None:

                    # The amount of padding that was added
                    pad_x = 0 if (opt.img_size[0]/frame.shape[1]) < (opt.img_size[1]/frame.shape[0]) else opt.img_size[0] - opt.img_size[1] / frame.shape[0] * frame.shape[1]
                    pad_y = 0 if (opt.img_size[0]/frame.shape[1]) > (opt.img_size[1]/frame.shape[0]) else opt.img_size[1] - opt.img_size[0] / frame.shape[1] * frame.shape[0]

                    # Image height and width after padding is removed
                    unpad_h = opt.img_size[1] - pad_y
                    unpad_w = opt.img_size[0] - pad_x

                    # Draw bounding boxes and labels of detections
                    # unique_classes = det[:, -1].cpu().unique()
                    #
                    # for i in unique_classes:
                    #     n = (det[:, -1].cpu() == i).sum()
                        # print('%g %ss' % (n, classes[int(i)]))

                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in det:
                        # Rescale coordinates to original dimensions
                        box_h = ((y2 - y1) / unpad_h) * frame.shape[0]
                        box_w = ((x2 - x1) / unpad_w) * frame.shape[1]
                        y1 = (((y1 - pad_y // 2) / unpad_h) * frame.shape[0]).round().item()
                        x1 = (((x1 - pad_x // 2) / unpad_w) * frame.shape[1]).round().item()
                        x2 = (x1 + box_w).round().item()
                        y2 = (y1 + box_h).round().item()
                        x1, y1, x2, y2 = max(x1, 1.0), max(y1, 1.0), min(x2, frame.shape[1]-1.0), min(y2, frame.shape[0]-1.0)

                        # label = '%s %.2f' % (classes[int(cls_pred)], conf)
                        # c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
                        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 1 / 3, [225, 255, 255], thickness=1,
                        #             lineType=cv2.LINE_AA)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[int(cls_pred)], 2)

                cv2.namedWindow("image", 0)
                cv2.resizeWindow("image", int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.8))
                cv2.imshow("image", frame)
                cv2.waitKey(1)

                    # cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
