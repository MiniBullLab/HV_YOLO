## data
root_dir = "/home/wfw/data/VOCdevkit/CityscapesFine/"
imgSize = (640, 352)
train_batch_size = 2
test_batch_size = 1
className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

## detect
net_config_path = "./cfg/shufflenetV2-0.5_spp_BerkeleyAll.cfg"
snapshotPath = "./snapshot/"
latest_weights_file = './snapshot/latest.pt'
best_weights_file = './snapshot/best.pt'
maxEpochs = 300

resume = False

optimType = 'poly'
base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
                     'lr': base_lr,
                     'momentum': 0.9,
                     'weight_decay': 5e-4}
             }
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
