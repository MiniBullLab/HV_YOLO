import argparse
import time

from models import *
from modelsShuffleNet import *
from data_loader import *
from utils.utils import *

from utils import torch_utils
import config.config as config

# Import test.py to get mAP after each epoch
import test

def train(
        net_config_path,
        data_config_path,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        weights_path='weights',
        report = False,
        multi_scale=False,
        freeze_bn=True,
        var=0,
):
    device = torch_utils.select_device()
    print("Using device: \"{}\"".format(device))
    img_size = [640, 352]

    if multi_scale:  # pass maximum multi_scale size
        img_size = 608
    else:
        torch.backends.cudnn.benchmark = True

    os.makedirs(weights_path, exist_ok=True)
    latest_weights_file = os.path.join(weights_path, 'latest.pt')
    best_weights_file = os.path.join(weights_path, 'best.pt')

    # Configure run
    data_config = parse_data_config(data_config_path)
    train_path = data_config['train']
    valPath = data_config["valid"]

    # Initialize model
    model = ShuffleYolo(net_config_path, img_size, freeze_bn=freeze_bn)

    # yoloLoss
    yoloLoss = []
    for m in model.module_list:
        for layer in m:
            if isinstance(layer, YoloLoss):
                yoloLoss.append(layer)

    # Get dataloader
    dataloader = ImageDetectTrainDataLoader(train_path, batch_size, img_size,
                                        multi_scale=multi_scale, augment=True, balancedSample=True)

    avg_loss = -1
    lr0 = 0.0002
    if resume:
        if torch.cuda.device_count() > 1:
            checkpoint = torch.load(latest_weights_file, map_location='cpu')
            state = convert_state_dict(checkpoint['model'])
            model.load_state_dict(state)
        else:
            checkpoint = torch.load(latest_weights_file, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9, weight_decay=5e-4)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            #best_mAP = checkpoint['best_mAP']

        del checkpoint  # current, saved

    else:
        start_epoch = 0
        best_mAP = -1

        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9, weight_decay=5e-4)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    model_info(model)
    t0 = time.time()
    for epoch in range(start_epoch, epochs):

        # Update scheduler (manual)  at 0, 54, 61 epochs to 1e-3, 1e-4, 1e-5
        if epoch > 70:
            lr = lr0 / 100
        elif epoch > 49: # 50 for no_balance_sample
            lr = lr0 / 10
        else:
            lr = lr0

        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(dataloader):
            # poly scheduler
            #lr = lr0 * (1 - (epoch * len(dataloader) + i) / (epochs * len(dataloader))) ** 0.9
            #for g in optimizer.param_groups:
            #    g['lr'] = lr

            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = lr0 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss = 0
            output = model(imgs.to(device))
            for k in range(0, 3):
                loss += yoloLoss[k](output[k], targets)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            if avg_loss < 0:
                avg_loss = (loss.cpu().detach().numpy() / batch_size)
            avg_loss = 0.9 * (loss.cpu().detach().numpy() / batch_size) + 0.1 * avg_loss
            print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch, i, len(dataloader), '%.3f' % avg_loss, '%.7f' % optimizer.param_groups[0]['lr'], time.time() - t0))
            t0 = time.time()

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      #'best_mAP': best_mAP,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest_weights_file)

        # Calculate mAP
        opt.imageFile = valPath
        opt.weights_path = 'weights/latest.pt'
        opt.img_size = img_size
        opt.cfg = net_config_path
        mAP, aps = test.main(opt)

        #if mAP >= best_mAP:
        #    best_mAP = mAP

        # Save best checkpoint
        #if mAP >= best_mAP:
        #    os.system('cp {} {}'.format(
        #        latest_weights_file,
        #        best_weights_file,
        #    ))

        # Save backup weights every 5 epochs
        if (epoch > 0) & (epoch % 1 == 0):
            backup_file_name = 'backup{}.pt'.format(epoch)
            backup_file_path = os.path.join(weights_path, backup_file_name)
            os.system('cp {} {}'.format(
                latest_weights_file,
                backup_file_path,
            ))

        # Write epoch results
        with open('results.txt', 'a') as file:
            #file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(config.className[i] + ": {:.3f} ".format(ap))
            file.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--data-config', type=str, default='cfg/coco.data', help='path to data config file')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--weights-path', type=str, default='weights', help='path to store weights')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--report', action='store_true', help='report TP, FP, FN, P and R per batch (slower)')
    parser.add_argument('--freeze', action='store_true', help='freeze darknet53.conv.74 layers for first epoch')

    opt = parser.parse_args()
    print(opt, end='\n\n')

    init_seeds()

    torch.cuda.empty_cache()
    train(
        opt.cfg,
        opt.data_config,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        weights_path=opt.weights_path,
        report=opt.report,
        multi_scale=opt.multi_scale,
        freeze_bn=opt.freeze,
    )
