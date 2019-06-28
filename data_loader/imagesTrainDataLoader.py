from .dataLoader import *
import config.config as config
from utils.utils import xyxy2xywh

class ImagesTrainDataLoader(DataLoader):
    def __init__(self, path, batch_size=1, img_size=[768, 320], \
                 multi_scale=False, augment=False, balancedSample=False):
        super().__init__(path)
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]

        if balancedSample:# MacOS (local)
            self.balancedFiles = {}
            self.nFs = {}
            for i in range(0, len(config.className)):
                with open("./data/classes/" + config.className[i] + ".txt", "r") as file:
                    self.balancedFiles[config.className[i]] = file.readlines()
                    self.balancedFiles[config.className[i]] = [path.replace('\n', '') for path in self.balancedFiles[config.className[i]]]
                    self.nFs[config.className[i]] = len(self.balancedFiles[config.className[i]])

        self.label_files = [path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]
        self.volid_label_det = [6, 7, 8]
        self.valid_label_det = [[0], [1], [2], [3], [4], [5], [9]]

        self.multi_scale = multi_scale
        self.augment = augment
        self.balancedSample = balancedSample

        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)#

        self.balancedCount = np.zeros(len(config.className))
        self.shuffled_vectors = {}
        if self.balancedSample:
            for i in range(0, len(config.className)):
                    self.shuffled_vectors[config.className[i]] = np.random.permutation(self.nFs[config.className[i]]) \
                        if self.augment else np.arange(self.nFs[config.className[i]])#
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        if self.balancedSample:
            randomLabel = np.random.randint(0, len(self.valid_label_det))
            print("loading labels {}".format(config.className[randomLabel]))
            balanced_file = self.balancedFiles[config.className[randomLabel]]
            ia = int(self.balancedCount[randomLabel])
            self.balancedCount[randomLabel] += self.batch_size
            ib = int(min(self.balancedCount[randomLabel], self.nFs[config.className[randomLabel]]))
            if self.balancedCount[randomLabel] > self.nFs[config.className[randomLabel]]:
                self.balancedCount[randomLabel] = 0
        else:
            ia = self.count * self.batch_size
            ib = min((self.count + 1) * self.batch_size, self.nF)

        # ia = self.count * self.batch_size
        # ib = min((self.count + 1) * self.batch_size, self.nF)

        if self.multi_scale:
            # Multi-Scale YOLO Training
            width = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            scale = float(self.width) / float(self.height)
            height = int(round(float(width / scale) / 32.0) * 32)
        else:
            # Fixed-Scale YOLO Training
            height = self.height
            width = self.width

        img_all = []
        labels_all = []
        for index, files_index in enumerate(range(ia, ib)):
            if self.balancedSample:
                img_path = balanced_file[self.shuffled_vectors[config.className[randomLabel]][files_index]]
                label_path = img_path.replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            else:
                img_path = self.img_files[self.shuffled_vector[files_index]]
                label_path = self.label_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            if img is None:
                continue

            augment_hsv = True
            if self.augment and augment_hsv:
                # SV augmentation by 50%
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            h, w, _ = img.shape
            img, ratio, padw, padh = self.resize_square(img, width=width, height=height, color=(127.5, 127.5, 127.5))

            # Load labels
            if os.path.isfile(label_path):
                labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)

                # Normalized xywh to pixel xyxy format
                labels0 = self.encode_label(labels0, self.volid_label_det, self.valid_label_det)
                labels = labels0.copy()
                labels[:, 1] = ratio * w * (labels0[:, 1] - labels0[:, 3] / 2) + padw // 2
                labels[:, 2] = ratio * h * (labels0[:, 2] - labels0[:, 4] / 2) + padh // 2
                labels[:, 3] = ratio * w * (labels0[:, 1] + labels0[:, 3] / 2) + padw // 2
                labels[:, 4] = ratio * h * (labels0[:, 2] + labels0[:, 4] / 2) + padh // 2
            else:
                labels = np.array([])

            # Augment image and labels
            if self.augment:
                img, labels, M = self.random_affine(img, labels, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.1))

            plotFlag = False
            if plotFlag:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10)) if index == 0 else None
                plt.subplot(4, 4, index + 1).imshow(img[:, :, ::-1])
                plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
                plt.axis('off')

            nL = len(labels)
            if nL > 0:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5].copy())
                labels[:, (1,3)] = labels[:, (1,3)]/width
                labels[:, (2,4)] = labels[:, (2,4)]/height

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    if nL > 0:
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip & (random.random() > 0.5):
                    img = np.flipud(img)
                    if nL > 0:
                        labels[:, 2] = 1 - labels[:, 2]

            delete_index = []
            for i, label in enumerate(labels):
                if label[2] + label[4]/2 > 1:
                    yold = label[2] - label[4] / 2
                    label[2] = (yold + float(1)) / float(2)
                    label[4] = float(1) - yold
                if label[3] < 0.005 or label[4] < 0.005:
                    delete_index.append(i)

            labels = np.delete(labels, delete_index, axis=0)

            img_all.append(img)
            labels_all.append(torch.from_numpy(labels))

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        return torch.from_numpy(img_all), labels_all

    def __len__(self):
        return self.nB  # number of batches