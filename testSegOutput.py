import time, cv2, os
from utility.utils import *
import scipy.misc as misc
from data_loader import *
from utility.torchModelProcess import TorchModelProcess

def segment(cfgPath, weights_path):
    torchModelProcess = TorchModelProcess()
    model = torchModelProcess.initModel(cfgPath, 0)
    model.setFreezeBn(True)
    torchModelProcess.loadLatestModelWeight(weights_path, model)
    torchModelProcess.modelTestInit(model)

    dataloader = ImagesLoader("/home/wfw/data/VOCdevkit/Tusimple/JPEGImages/", batch_size=1, img_size=(640, 352))
    prev_time = time.time()
    for i, (img_paths, img) in enumerate(dataloader):
        # Get detections
        with torch.no_grad():
            output = model(torch.from_numpy(img).unsqueeze(0).to(torchModelProcess.getDevice()))

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        prev_time = time.time()

        # oriImage shape, input shape
        oriImage = cv2.imread(img_paths[0])
        ori_w, ori_h = oriImage.shape[1], oriImage.shape[0]
        pre_h, pre_w = 352, 640

        # segmentation
        segmentations = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
        ratio = min(float(pre_w) / ori_w, float(pre_h) / ori_h)  # ratio  = old / new
        new_shape = [round(ori_h * ratio), round(ori_w * ratio)]
        dw = pre_w - new_shape[1]  # width padding
        dh = pre_h - new_shape[0]  # height padding
        padH = [dh // 2, dh - (dh // 2)]
        padW = [dw // 2, dw - (dw // 2)]
        segmentations = segmentations[padH[0]:pre_h - padH[1], padW[0]:pre_w - padW[1]]
        segmentations = segmentations.astype(np.float32)
        segmentations = misc.imresize(segmentations, [ori_h, ori_w], 'nearest',
                                      mode='F')  # float32 with F mode, resize back to orig_size
        decoded = decode_segmap(oriImage, segmentations, 2)

        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(decoded.shape[1] * 0.5), int(decoded.shape[0] * 0.5))
        cv2.imshow('image', decoded)

        if cv2.waitKey() & 0xFF == 27:
            break
