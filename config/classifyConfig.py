# data
imgSize = (32, 32)
train_batch_size = 16
test_batch_size = 1
# className = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

# classify
snapshotPath = "./snapshot/"
latest_weights_file = './snapshot/latest.pt'
best_weights_file = './snapshot/best.pt'
maxEpochs = 100

base_lr = 2e-4
lr_power = 0.9
optimizerConfig = {0: {'optimizer': 'SGD',
                       'momentum': 0.9,
                       'weight_decay': 5e-4}
                   }
accumulated_batches = 1

display = 20
