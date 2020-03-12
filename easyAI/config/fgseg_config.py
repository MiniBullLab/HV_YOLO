from easyAI.config.base_config import *

image_size = (400, 500)

train_batch_size = 1
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "seg_latest.h5")
best_weights_file = os.path.join(snapshotPath, "seg_best.h5")
save_image_dir = os.path.join(root_save_dir, "seg_img")
maxEpochs = 100
