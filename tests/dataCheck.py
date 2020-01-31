import numpy as np
import cv2
from easy_ai.data_loader import ImageDetectTrainDataLoader


class dataCheck():
    def __init__(self):
        pass

    def decode_labels(self, img, labels):

        h, w, _ = img.shape

        x1 = w * (labels[1] - labels[3]/2)
        y1 = h * (labels[2] - labels[4]/2)
        x2 = w * (labels[1] + labels[3]/2)
        y2 = h * (labels[2] + labels[4]/2)

        return x1, y1, x2, y2

    def detectData(self):
        # Get dataloader
        color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
        dataloader = ImageDetectTrainDataLoader("/home/wfw/data/VOCdevkit/BerkeleyDet/ImageSets/train.txt", batchSize=1,
                                                imageSize=[640, 352], multi_scale=False, augment=False, balancedSample=False)

        for i, (imgs, labels) in enumerate(dataloader):
            for img, label in zip(imgs, labels):
                print("Image: {}".format(i))
                img = img.numpy()
                img = np.transpose(img, (1, 2, 0)).copy()
                label = label.numpy()
                for l in label:
                    xmin, ymin, xmax, ymax = self.decode_labels(img, l)
                    # print("w for obj: {}, h for obj: {}".format((xmax-xmin) / 640 * 1280, (ymax-ymin) / 352 * 720))
                    # if ((xmax-xmin) / 640 * 1280) < 6.8 or ((ymax-ymin) / 352 * 720) < 5.0:
                    #     print("w for obj: {}, h for obj: {}".format((xmax - xmin) / 640 * 1280, (ymax - ymin) / 352 * 720))
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color[int(l[0])], 2)

                cv2.imshow('img', img)
                key = cv2.waitKey()
            if key == 27:
                break

if __name__ == "__main__":
    checkData = dataCheck()
    checkData.detectData()