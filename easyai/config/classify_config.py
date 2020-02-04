from easyai.config.base_config import *

TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# data
imgSize = (32, 32)
train_batch_size = 8
test_batch_size = 1

# classify
log_name = "classify"
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, 'latest.pt')
best_weights_file = os.path.join(snapshotPath, 'best.pt')
maxEpochs = 100

base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
                       'momentum': 0.9,
                       'weight_decay': 5e-4}
                   }
accumulated_batches = 1

enable_mixed_precision = False

display = 20