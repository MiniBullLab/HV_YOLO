# select architecture
#arch = "yolov3SPP"
cfg = "cfg/darknet53_mask.cfg"
data_config_path = "cfg/berkeley.data"
net_config_path = "cfg/shufflenetV2-0.5_spp_BerkeleyAll.cfg"

# pretrained model
pretainModel = None #"./weights/model_best.pth"

# resume from checkpoint
resume = "./weights/model_best.pth" #None #"./weights/best.pt"#"./snapshot/yolov3-SPP_best_model.pkl"#None#

# learning rate
base_lr = 5e-4
lr_decay_epochs = [120,160]
lr_decay = 0.1

maxEpochs =500

# L2 regularizer
optimizer = 'SGD'
lr_power = 0.9
momentum = 0.9
weight_decay = 5e-3
accumulated_batches = 1

snapshot = 2
snapshot_path = "snapshot"

# dataSet
#imdb_train = "/home/lipj/Car_identify/data/DayData/train/"
#imdb_val = "/home/lipj/Car_identify/data/DayData/val/"
imgSize = [640, 352]
trainList = "/home/sugon/darknetV3-master/data/train.txt"
valList = "/home/sugon/darknetV3-master/data/test.txt"
valLabelPath = "/home/sugon/data/VOCdevkit/BerkeleyDet/Annotations/"
train_batch_size = 32
test_batch_size = 1

# yolo param
anchors = [[[44,76], [65,116], [117,164]],
           [[15,18], [22,38], [29,56]],
           [[7,10], [9,18], [14,30]]]
#className = ['truck', 'person', 'bicycle', 'car', 'motorbike', 'bus']
className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']#['car', 'person', 'motor']
numClasses = 7
iouThresh = 0.45
confThresh=0.24

jitter = 0.3
#hue = 0.1
hue = 0
saturation = 1.5
exposure = 1.5

# GPU SETTINGS
CUDA_DEVICE = [0] # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.

# SETTINGS FOR DISPLAYING ON TENSORBOARD
visdomTrain = False
visdomVal = False
