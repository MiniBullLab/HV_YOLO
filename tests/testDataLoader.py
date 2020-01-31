import numpy as np
from easy_ai.data_loader import ImageClassifyTrainDataLoader
from easy_ai.config import classifyConfig

imageList = open("/home/wfw/data/VOCdevkit/cifar10/train_data.txt", "r")
batch_size = len(imageList.readlines())
dataloader = ImageClassifyTrainDataLoader("/home/wfw/data/VOCdevkit/cifar10/train_data.txt",
                                          batch_size=batch_size,
                                          img_size=classifyConfig.imgSize)
# total_images = len(dataloader)
train = iter(dataloader).__next__()[0]
mean = np.mean(train.numpy(), axis=(0, 2, 3))
std = np.std(train.numpy(), axis=(0, 2, 3))
print(mean, std)