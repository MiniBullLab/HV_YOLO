from easyAI.config.base_config import *

# data
imgSize = (32, 32)
train_batch_size = 16
test_batch_size = 1
# className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

# classify
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "cls_latest.pt")
best_weights_file = os.path.join(snapshotPath, "cls_best.pt")
maxEpochs = 100

base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
                       'momentum': 0.9,
                       'weight_decay': 5e-4}
                   }
accumulated_batches = 1

display = 20
