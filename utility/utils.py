import random
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from collections import OrderedDict

import matplotlib.pyplot as plt

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

def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

#################################################################################################
def multiLrMethod(model, hyperparams):
    stopLayer = "data"
    for k, v in hyperparams.items():
        if ("conv" in k and v!='None') or ("innerProduct"in k and v!='None'):
            stopLayer = k
            stopType = v
    if stopLayer != "data":
        if stopType == "finetune":
            print(stopLayer)
            for name, p in model.named_parameters():
                #for layerName in stopLayer:
                if stopLayer in name:
                    p.requires_grad = True
                    break
                else:
                    p.requires_grad = False

            # define optimizer
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        elif stopType == "diffLRate":
            base_params = []
            for name, p in model.named_parameters():
                #for layerName in stopLayer:
                if stopLayer in name:
                    base_params.append(p)
                else:
                    break

            base_layer_params = list(map(id, base_params))
            special_layers_params = filter(lambda p: id(p) not in base_layer_params, model.parameters())

            # define optimizer
            optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': special_layers_params, 'lr': 1e-2}], lr=1e-3)

        elif stopType == "freeze":
            bn_layers = []
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_layers.append(m)

            layer_i = 0
            for name, p in model.named_parameters():
                #for layerName in stopLayer:
                if stopLayer in name:
                    break
                #elif name.split('.')[5] == "bn" in name and name.split('.')[6] == "weight":
                elif "bn" in name and "weight" in name:
                    bn_layers[layer_i].momentum=0
                    layer_i += 1

            # define optimizer
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)


    return model, optimizer

class vis_bn():
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function

    descript:
    # 1. visual params from bn layer
    # 2. visual featuremap from con layer
    # 3. visual low dim of the last layer
    """

    def __init__(self, model):
        self.axss = []
        self.bn_layers = []
        self.bn_layer_name = []
        bn_layer_index = 0

        for name, p in model.named_parameters():
            if 'bn' in name and 'weight' in name:
                self.bn_layer_name.append('.'.join(name.split(".")[:-1]))

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                f, axs = plt.subplots(4, 1, figsize=(6, 6))
                f.suptitle(self.bn_layer_name[bn_layer_index])
                bn_layer_index += 1
                self.axss.append(axs)
                self.bn_layers.append(m)

    def plot(self):
        for i, axs in enumerate(self.axss):
            m = self.bn_layers[i]
            self.plot_hist(axs, m.weight.data, m.bias.data, m.running_mean.data, m.running_var.data)

    def plot_hist(self, axs, weight, bias, running_mean, running_var):
        [a.clear() for a in [axs[0], axs[1], axs[2], axs[3]]]
        axs[0].bar(range(len(running_mean.cpu().numpy())), weight.cpu().numpy(), color='#FF9359')
        axs[1].bar(range(len(running_var.cpu().numpy())), bias.cpu().numpy(), color='g')
        axs[2].bar(range(len(running_mean.cpu().numpy())), running_mean.cpu().numpy(), color='#74BCFF')
        axs[3].bar(range(len(running_var.cpu().numpy())), running_var.cpu().numpy(), color='y')
        axs[0].set_ylabel('weight')
        axs[1].set_ylabel('bias')
        axs[2].set_ylabel('running_mean')
        axs[3].set_ylabel('running_var')
        plt.pause(0.01)