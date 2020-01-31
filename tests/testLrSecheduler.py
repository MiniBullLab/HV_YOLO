import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from easy_ai.solver.lr_scheduler import PolyLR
import matplotlib.pyplot as plt


# Visualize
def show_graph(lr_lists, epochs, steps, out_name):
    plt.clf()
    plt.rcParams['figure.figsize'] = [20, 5]
    x = list(range(epochs * steps))
    plt.plot(x, lr_lists, label=out_name)
    plt.plot()

    plt.ylim(10e-7, 1)
    plt.yscale("log")
    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.title("plot learning rate secheduler {}".format(out_name))
    plt.legend()
    plt.show()


def main():
    print("process start...")
    baseLr = 1e-3
    endLr = 1e-5
    model = nn.Linear(1280, 1000)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
                                         momentum=0.9, weight_decay=5e-4)
    nepoches = 100
    nImages = 2000
    lrName = "multiLR"

    lrList = []
    # LrScheduler = WarmupMultiStepLR(baseLr, [[50, 1], [70, 0.1], [100, 0.01]])
    # LrScheduler = MultiStageLR(baseLr, [[50, 1], [70, 0.1], [100, 0.01]])
    LrScheduler = PolyLR(baseLr, nepoches * nImages)
    #LrScheduler = WarmupPolyLR(baseLr, nepoches * nImages)
    # LrScheduler = CosineLR(baseLr, nepoches * nImages)
    # LrScheduler = WarmupCosineLR(baseLr, nepoches * nImages)
    # LrScheduler = LinearIncreaseLR(baseLr, endLr, nepoches * nImages)
    # LrScheduler = WarmupLinearIncreaseLR(baseLr, endLr, nepoches * nImages)

    for epoch in range(0, nepoches):
        for idx in range(0, nImages):
            current_idx = epoch * nImages + idx
            lr = LrScheduler.get_lr(epoch, current_idx)
            LrScheduler.adjust_learning_rate(optimizer, lr)
            lrList.append(optimizer.param_groups[0]['lr'])

    show_graph(lrList, nepoches, nImages, lrName)

    print("process end!")


if __name__ == '__main__':
    main()

