import os

root_save_dir = "./log"
model_save_dir_name = "snapshot"

# data
train_set = "/home/lpj/github/data/super/train"
val_set = "/home/lpj/github/data/super/test"
train_batch_size = 2
test_batch_size = 1

in_nc=1
crop_size = 72
upscale_factor = 3

# detect
snapshotPath = os.path.join(root_save_dir, model_save_dir_name)
latest_weights_file = os.path.join(snapshotPath, 'latest.pt')
best_weights_file = os.path.join(snapshotPath, 'best.pt')
maxEpochs = 100

base_lr = 1e-3
optimizerConfig = {0: {'optimizer': 'Adam'}
                  }
accumulated_batches = 1

enable_mixed_precision = False

display = 20

# speed
runType = "video"
testImageFolder = "/home/wfw/HASCO/data/image/"
testVideoFile = "/home/wfw/HASCO/data/video/VIDEO-5.MPG"
weightPath = "./weights/backup74.pt"
confThresh = 0.5
nmsThresh = 0.45
bnFuse = True