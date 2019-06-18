import argparse
import time, cv2

from models import *
from utils.datasets import *
from utils.utils import *

def img_process(img_path, img_size):
    width, height = img_size
    # Read image
    img = cv2.imread(img_path)  # BGR

    # Padded resize
    img, _, _, _ = resize_square(img, width=width, height=height, color=(127.5, 127.5, 127.5))

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    # img -= self.rgb_mean
    # img /= self.rgb_std
    img /= 255.0

    return img

def networkInit(cfg, img_size, weights_path):
    # Load model
    model = Darknet(cfg, img_size)
    print("Loading Model from {}......".format(weights_path))
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    del checkpoint

    model.to(0).eval()

    return model

def detect(model, img_path, img_size, labels, conf_thres):
    img = img_process(img_path, img_size)

    # Get detections
    detections = None
    with torch.no_grad():
        pred = model(torch.from_numpy(img).unsqueeze(0).to(0))
        pred = pred[pred[:, :, 4] > conf_thres]

        if len(pred) > 0:
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, 0.45)

    oriImg = cv2.imread(img_path)
    # The amount of padding that was added
    pad_x = 0 if (img_size[0]/oriImg.shape[1]) < (img_size[1]/oriImg.shape[0]) else img_size[0] - img_size[1] / oriImg.shape[0] * oriImg.shape[1]
    pad_y = 0 if (img_size[0]/oriImg.shape[1]) > (img_size[1]/oriImg.shape[0]) else img_size[1] - img_size[0] / oriImg.shape[1] * oriImg.shape[0]
    # Image height and width after padding is removed
    unpad_h = img_size[1] - pad_y
    unpad_w = img_size[0] - pad_x

    detects = []
    if detections is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * oriImg.shape[0]
            box_w = ((x2 - x1) / unpad_w) * oriImg.shape[1]
            y1 = (((y1 - pad_y // 2) / unpad_h) * oriImg.shape[0]).round().item()
            x1 = (((x1 - pad_x // 2) / unpad_w) * oriImg.shape[1]).round().item()
            x2 = (x1 + box_w).round().item()
            y2 = (y1 + box_h).round().item()
            x1, y1, x2, y2 = int(max(x1, 1.0)), int(max(y1, 1.0)), int(max(x2, 0)), int(max(y2, 0))
            detects.append(dict(label_name=labels[int(cls_pred)], box=[x1, y1, x2, y2], score = float(cls_conf * conf)))

    return detects

if __name__ == '__main__':
    model = networkInit("./cfg/yolov3-spp.cfg", [768, 320], "./weights/best.pt")
    detects = detect(model, "/home/wfw/data/VOCdevkit/BerkeleyDet/JPEGImages/b1d0a191-03dcecc2.jpg", [768, 320], 0.24)
    print(detects)
