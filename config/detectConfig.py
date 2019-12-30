from config.base_config import *

# data
imgSize = (640, 352)
train_batch_size = 2
test_batch_size = 1
className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

# detect
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "det_latest.pt")
best_weights_file = os.path.join(snapshotPath, "det_best.pt")
maxEpochs = 300

base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
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
