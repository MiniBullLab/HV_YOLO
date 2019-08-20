import random
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import config.config as config
import xml.etree.ElementTree as ET
from collections import OrderedDict

from .utils import torch_utils

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    import torch
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # histogram of occurrences per class
    import glob

    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


def plot_results():
    # Plot YOLO training results file 'results.txt'
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    # import os; os.system('rm -rf results.txt && wget https://storage.googleapis.com/ultralytics/results_v1_0.txt')

    plt.figure(figsize=(16, 8))
    s = ['X', 'Y', 'Width', 'Height', 'Objectness', 'Classification', 'Total Loss', 'Precision', 'Recall', 'mAP']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 7, 8, 17, 18, 16]).T  # column 16 is mAP
        n = results.shape[1]
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.plot(range(1, n), results[i, 1:], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()