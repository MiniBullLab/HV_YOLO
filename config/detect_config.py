# data
imgSize = (640, 352)
train_batch_size = 2
test_batch_size = 1
className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

# detect
log_name = "detect"
snapshotPath = "./snapshot/"
latest_weights_file = './snapshot/model_epoch_62.pt'
best_weights_file = './snapshot/best.pt'
maxEpochs = 300

base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
                       'momentum': 0.9,
                       'weight_decay': 5e-4}
                   }
accumulated_batches = 1

display = 20

# speed
runType = "video"
testImageFolder = "/home/wfw/HASCO/data/image/"
testVideoFile = "/home/wfw/HASCO/data/video/VIDEO-5.MPG"
weightPath = "./weights/backup74.pt"
confThresh = 0.5
nmsThresh = 0.45
bnFuse = True
