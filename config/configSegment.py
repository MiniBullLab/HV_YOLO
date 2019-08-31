## data
imgSize = (640, 352)
trainPath = "/home/lpj/github/data/Tusimple/ImageSets/train.txt"
valPath = "/home/lpj/github/data/Tusimple/ImageSets/val.txt"
# class_path = "./data/berkeley.names"
train_batch_size = 16
test_batch_size = 1
className = ['background', 'lane']

## detect
net_config_path = "./cfg/mobileFCN.cfg"
snapshotPath = "./snapshot/"
maxEpochs = 300

pretainModel = "./weights/mobilenetv2_best.pt"
resume = None #"./snapshot/latest.pt"

optimType = 'poly'
base_lr = 1e-3
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