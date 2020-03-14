from easyai.config.base_config import *

# data
imgSize = (440, 512)
train_batch_size = 1
test_batch_size = 1

label_is_gray = True
className = [('background', '255'),
             ('lane', '0')]

# segment
log_name = "segment"
save_evaluation_path = os.path.join(root_save_dir, 'seg_evaluation.txt')

is_save_epoch_model = True
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, 'latest.pt')
best_weights_file = os.path.join(snapshotPath, 'best.pt')
maxEpochs = 300

base_lr = 1e-2
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'RMSprop',
                       'alpha': 0.9,
                       'eps': 1e-08,
                       'weight_decay': 0.}
                   }
accumulated_batches = 1

enable_freeze_layer = False
freeze_layer_name = "route_0"

enable_mixed_precision = False

display = 20

# speed
confThresh = 0.5
nmsThresh = 0.45
