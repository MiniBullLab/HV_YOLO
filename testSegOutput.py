import argparse
import time, cv2, os

import torch
import torch.nn as nn
from utility.utils import *
import scipy.misc as misc

from data_loader import *
# from models.bisenet import BiSeNet
from model.modelParse import ModelParse
from config import config
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
f_path = os.path.dirname(os.path.realpath(__file__)) + '/'

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def decode_segmap(pic, detections, nClass):
	colors = [  # [  0,   0,   0],
			[128, 64, 128],
			[244, 35, 232],
			[70, 70, 70],
			[102, 102, 156],
			[190, 153, 153],
			[153, 153, 153],
			[250, 170, 30],
			[220, 220, 0],
			[107, 142, 35],
			[152, 251, 152],
			[0, 130, 180],
			[220, 20, 60],
			[255, 0, 0],
			[0, 0, 142],
			[0, 0, 70],
			[0, 60, 100],
			[0, 80, 100],
			[0, 0, 230],
			[119, 11, 32]]

	label_colours = dict(zip(range(19), colors))
	img = cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2BGR)  # convert PIL.image to cv2.mat

	r = detections.copy()
	g = detections.copy()
	b = detections.copy()
	for l in range(0, nClass):
		r[detections == l] = label_colours[l][0]
		g[detections == l] = label_colours[l][1]
		b[detections == l] = label_colours[l][2]

	rgb = np.zeros((detections.shape[0], detections.shape[1], 3))

	rgb[:, :, 0] = (r * 0.4 + img[:,:,2] * 0.6) / 255.0
	rgb[:, :, 1] = (g * 0.4 + img[:,:,1] * 0.6) / 255.0
	rgb[:, :, 2] = (b * 0.4 + img[:,:,0] * 0.6) / 255.0

	return rgb

# model select
# model = BiSeNet(config.num_classes, is_training=False, criterion=None, ohem_criterion=None)
modelParse = ModelParse()
model = modelParse.parse("./cfg/mobileFCN.cfg")

if torch.cuda.device_count() > 1:
	checkpoint = convert_state_dict(torch.load("/home/wfw/HASCO/MultiTask/MTSD/snapshot/Shuffle_BiseNet_best_model_0.566.pkl", map_location='cpu')['model_state'])
	model.load_state_dict(checkpoint)
else:
	# checkpoint = torch.load("/home/wfw/HASCO/MultiTask/MTSD/snapshot/Shuffle_BiseNet_best_model_0.566.pkl", map_location='cpu')
	checkpoint = torch.load("./weights/mobileFCN.pt", map_location='cpu')
	model.load_state_dict(checkpoint['model'])
	print("{}: IoU is {}".format(checkpoint['epoch'], checkpoint['best_mIoU']))
del checkpoint

model.to(device).eval()

# Set Dataloader
dataloader = ImagesLoader("/home/wfw/data/VOCdevkit/Tusimple/JPEGImages/", batch_size=1, img_size=[640, 352])
prev_time = time.time()

for i, (img_paths, img) in enumerate(dataloader):

	# Get detections
	with torch.no_grad():
		output = model(torch.from_numpy(img).unsqueeze(0).to(device))

	print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
	prev_time = time.time()

	# oriImage shape, input shape
	oriImage = cv2.imread(img_paths[0])
	ori_w, ori_h = oriImage.shape[1], oriImage.shape[0]
	pre_h, pre_w = 352, 640

	# segmentation
	segmentations = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
	ratio = min(float(pre_w) / ori_w, float(pre_h) / ori_h)  # ratio  = old / new
	new_shape = [round(ori_h * ratio), round(ori_w * ratio)]
	dw = pre_w - new_shape[1]  # width padding
	dh = pre_h - new_shape[0]  # height padding
	padH = [dh // 2, dh - (dh // 2)]
	padW = [dw // 2, dw - (dw // 2)]
	segmentations = segmentations[padH[0]:pre_h - padH[1], padW[0]:pre_w - padW[1]]
	segmentations = segmentations.astype(np.float32)
	segmentations = misc.imresize(segmentations, [ori_h, ori_w], 'nearest',
								  mode='F')  # float32 with F mode, resize back to orig_size
	decoded = decode_segmap(oriImage, segmentations, 2)

	cv2.namedWindow("image", 0)
	cv2.resizeWindow("image", int(decoded.shape[1] * 0.5), int(decoded.shape[0] * 0.5))
	cv2.imshow('image', decoded)

	if cv2.waitKey() & 0xFF == 27:
		break
