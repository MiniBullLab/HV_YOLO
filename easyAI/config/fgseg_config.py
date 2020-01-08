from easyAI.config.base_config import *

image_size = (440, 512)

train_batch_size = 1
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "seg_latest.pt")
best_weights_file = os.path.join(snapshotPath, "seg_best.pt")
maxEpochs = 100
