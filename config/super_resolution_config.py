## data
train_set = "/home/lpj/github/data/super/train"
val_set = "/home/lpj/github/data/super/test"
train_batch_size = 2
test_batch_size = 1

in_nc=1
crop_size = 72
upscale_factor = 3

## detect
net_config_path = " "
snapshotPath = "./snapshot/"
latest_weights_file = './snapshot/latest.pt'
best_weights_file = './snapshot/best.pt'
maxEpochs = 100

base_lr = 1e-3
optimizerConfig = {0: {'optimizer': 'Adam'}
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