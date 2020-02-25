import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tasks.seg.segment_train import SegmentionTrain
from easyai.data_loader.seg.segment_dataloader import get_segment_train_dataloader
from easyai.config import segment_config
from easyai.solver.lr_scheduler import PolyLR
from easyai.solver.torch_optimizer import TorchOptimizer
from easyai.tools.model_summary import summary

def main():
    segment_train = SegmentionTrain("../cfg/seg/fgsegv2.cfg", 0)
    segment_train.torchOptimizer.freeze_front_layer(segment_train.model, "route_0")
    summary(segment_train.model, [1, 3, 512, 440])

    dataloader = get_segment_train_dataloader("/home/wfw/data/VOCdevkit/LED/ImageSets/train.txt", segment_config.imgSize,
                                              segment_config.train_batch_size)
    segment_train.total_images = len(dataloader)
    segment_train.load_param(segment_config.latest_weights_file)

    # set learning policy
    total_iteration = segment_config.maxEpochs * len(dataloader)
    polyLR = PolyLR(segment_config.base_lr, total_iteration, segment_config.lr_power)

    segment_train.timer.tic()
    for epoch in range(segment_train.start_epoch, segment_config.maxEpochs):
        # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
        segment_train.optimizer.zero_grad()
        for idx, (images, segments) in enumerate(dataloader):
            current_idx = epoch * segment_train.total_images + idx
            lr = polyLR.get_lr(epoch, current_idx)
            polyLR.adjust_learning_rate(segment_train.optimizer, lr)
            loss = segment_train.compute_backward(images, segments, idx)
            segment_train.update_logger(idx, segment_train.total_images, epoch, loss.data)

        save_model_path = segment_train.save_train_model(epoch)
        segment_train.test("/home/wfw/data/VOCdevkit/LED/ImageSets/val.txt", epoch, save_model_path)

if __name__ == '__main__':
    main()