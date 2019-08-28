## data
root_dir =  "/home/wfw/data/VOCdevkit/CityscapesFine/"
imgSize = (640, 352)
trainPath = "/home/wfw/data/VOCdevkit/BerkeleyDet/ImageSets/train.txt"
valPath = "/home/wfw/data/VOCdevkit/BerkeleyDet/ImageSets/test.txt"
# class_path = "./data/berkeley.names"
train_batch_size = 16
test_batch_size = 1
className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

## detect
net_config_path = "./cfg/shufflenetV2-0.5_spp_BerkeleyAll.cfg"
snapshotPath = "./snapshot/"
maxEpochs = 300

resume = True #"./snapshot/latest.pt"

optimType = 'poly'
base_lr = 2e-4
lr_power = 0.9
momentum = 0.9
weight_decay = 5e-4
accumulated_batches = 1

display = 20

## speed
runType = "video"
testImageFolder = "/home/wfw/HASCO/data/image/"
testVideoFile = "/home/wfw/HASCO/data/video/VIDEO-5.MPG"
weightPath = "./weights/backup74.pt"
confThresh = 0.5
nmsThresh = 0.45
bnFuse = True

## train
# detection = False
# segmentation = True
# det_num_class = 7
# seg_num_class = 19
#
# net_config_path = "./cfg/shuffleMTSD.cfg"
# snapshotPath = "./snapshot/"
# maxEpochs = 300
#
# # model
# pretainModel = None # "./shuffleMTSD.pth"
# resume = "./snapshot/latest.pt"
#
# optimType = 'poly'
# base_lr = 1e-3
# lr_power = 0.9
# momentum = 0.9
# weight_decay = 5e-4
# accumulated_batches = 1
#
# net_size = 1 # net size for shufflenetV2
# CUDA_DEVICE = [0]
#
# # evalute in train process
# evaluate = True
#
# # freeze param
# freezeParam = True
# finetuneParam = False
# finetuneRate = 0.3
# detRate = 0
# segRate = 1
# warm_up = False
#
# ## eval
# image_folder = root_dir + "/JPEGImages/val/"
# output_path = "./results/"
# annotation_path = root_dir + "Annotations/val/"
# className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']
# # ["truck", "person", "bicycle", "car", "motorbike", "bus"]
# nms_thres = 0.45
#
# ## speed
# runType = "video"
# testImageFolder = "/home/wfw/HASCO/data/image/"
# testVideoFile = "/home/wfw/HASCO/data/video/VIDEO-5.MPG"
# weightPath = "./snapshot/best.pt"
# confThresh = 0.5
# bnFuse = True
#
# #output flag
# detOutput = True
# segOutput = False
# segLineOutput = True
# lablemap = "./docs/segCityscapesclass.json"