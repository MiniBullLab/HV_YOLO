from easyAI.config.base_config import *

image_size = (512, 440)

train_batch_size = 1
snapshotPath = os.path.join(root_save_dir, model_save_dir)
latest_weights_file = os.path.join(snapshotPath, "seg_latest.pt")
best_weights_file = os.path.join(snapshotPath, "seg_best.pt")
save_image_dir = os.path.join(root_save_dir, "seg_img")
maxEpochs = 100
