from config.base_config import *


# data
imgSize = (640, 352)
train_batch_size = 16
test_batch_size = 1
className = ['background', 'lane']

# seg
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "seg_latest.pt")
best_weights_file = os.path.join(snapshotPath, "seg_best.pt")
maxEpochs = 300

base_lr = 1e-3
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