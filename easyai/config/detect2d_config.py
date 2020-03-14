from easyai.config.base_config import *

# data
imgSize = (640, 352)
train_batch_size = 2
test_batch_size = 1
className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

# detect
log_name = "detect"
save_result_dir = os.path.join(root_save_dir, 'det_results')
save_evaluation_path = os.path.join(root_save_dir, 'det2d_evaluation.txt')

is_save_epoch_model = True
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, 'latest.pt')
best_weights_file = os.path.join(snapshotPath, 'best.pt')
maxEpochs = 300

base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
                       'momentum': 0.9,
                       'weight_decay': 5e-4}
                   }
accumulated_batches = 1

enable_mixed_precision = False

display = 20

# speed
confThresh = 0.5
nmsThresh = 0.45


class Detect2dConfig(BaseConfig):

    def __init__(self):
        pass
