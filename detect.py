import argparse
import time

from modelsShuffleNet import *
from utils.datasets import *
from utils.utils import *

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

parser = argparse.ArgumentParser()
# Get data configuration

parser.add_argument('-image_folder', type=str, default='data/samples', help='path to images')
parser.add_argument('-weight_name', type=str, default='yolov3.pt', help='path to your model')
parser.add_argument('-output_folder', type=str, default='output', help='path to outputs')
parser.add_argument('-draw_flag', type=bool, default=False)
parser.add_argument('-plot_flag', type=bool, default=True)
parser.add_argument('-txt_out', type=bool, default=False)

parser.add_argument('-cfg', type=str, default=f_path + 'cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-class_path', type=str, default=f_path + 'data/berkeley.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.50, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
opt = parser.parse_args()
print(opt)


def main(opt):
    opt.img_size = [640, 352]
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
        if torch.cuda.device_count() > 1:
            checkpoint = convert_state_dict(torch.load(weights_path, map_location='cpu')['model'])
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        del checkpoint

    model.to(device).eval()

    # Set Dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    dataloader = load_images(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    #timeCal = 0
    #numCal = 0
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
            pred = pred[pred[:, :, 4] > opt.conf_thres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), opt.conf_thres, opt.nms_thres)
                img_detections.extend(detections)
                imgs.extend(img_paths)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        #timeCal += time.time() - prev_time
        #numCal += 1
        prev_time = time.time()

    #print("average time for per image: {}".format(timeCal / numCal))
    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if len(img_detections) == 0:
        return

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

        if opt.plot_flag:
            img = cv2.imread(path)

        # The amount of padding that was added
        pad_x = 0 if (opt.img_size[0]/img.shape[1]) < (opt.img_size[1]/img.shape[0]) else opt.img_size[0] - opt.img_size[1] / img.shape[0] * img.shape[1]
        pad_y = 0 if (opt.img_size[0]/img.shape[1]) > (opt.img_size[1]/img.shape[0]) else opt.img_size[1] - opt.img_size[0] / img.shape[1] * img.shape[0]

        # Image height and width after padding is removed
        unpad_h = opt.img_size[1] - pad_y
        unpad_w = opt.img_size[0] - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_img_path = os.path.join(opt.output_folder, path.split('/')[-1])
            results_txt_path = results_img_path + '.txt'
            if os.path.isfile(results_txt_path):
                os.remove(results_txt_path)

            for i in unique_classes:
                n = (detections[:, -1].cpu() == i).sum()
                print('%g %ss' % (n, classes[int(i)]))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round().item()
                x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                x2 = (x1 + box_w).round().item()
                y2 = (y1 + box_h).round().item()
                x1, y1, x2, y2 = max(x1, 1.0), max(y1, 1.0), min(x2, img.shape[1]-1.0), min(y2, img.shape[0]-1.0)
                if opt.draw_flag:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), [0,255,0], 2)

                # write to file
                if opt.txt_out:
                    with open(results_txt_path, 'a') as file:
                        file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, cls_pred, cls_conf * conf))

                if opt.plot_flag:
                    # Add the bbox to the plot
                    label = '%s %.2f' % (classes[int(cls_pred)], conf)
                    color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                    plot_one_box([x1, y1, x2, y2], img, label=label, color=color)

        if opt.draw_flag:
            cv2.namedWindow("image", 0)
            cv2.resizeWindow("image", int(img.shape[1] * 0.8), int(img.shape[0] * 0.8))
            cv2.imshow("image", img)
            cv2.waitKey()

        if opt.plot_flag:
            # Save generated image with detections
            cv2.imwrite(results_img_path.replace('.bmp', '.jpg').replace('.tif', '.jpg'), img)

    if platform == 'darwin':  # MacOS (local)
        os.system('open ' + opt.output_folder)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
