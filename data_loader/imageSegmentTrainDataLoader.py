from .dataLoader import *
from .trainDataProcess import TrainDataProcess
from PIL import Image

multi_scale = False

class ImageSegmentTrainDataLoader(DataLoader):
    def __init__(self, path, batch_size=1, img_size=(768, 320), augment=False):
        super().__init__(path)

        self.trainDataProcess = TrainDataProcess()
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]
        self.seg_files = [path.replace('JPEGImages', 'SegmentLabel').replace('.png', '.png').replace('.jpg', '.png') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.imageSize = img_size
        self.color = (127.5, 127.5, 127.5)

        self.volid_label_seg = []
        self.valid_label_seg = [[0], [1]]

        self.augment = augment

        assert self.nB > 0, 'No images found in path %s' % path

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        if multi_scale:
            # Multi-Scale YOLO Training
            width = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            scale = float(self.imageSize[0]) / float(self.imageSize[1])
            height = int(round(float(width / scale) / 32.0) * 32)
        else:
            # Fixed-Scale YOLO Training
            width = self.imageSize[0]
            height = self.imageSize[1]

        img_all = []
        seg_all = []
        for index, files_index in enumerate(range(ia, ib)):
            img_path = self.img_files[self.shuffled_vector[files_index]]
            seg_path = self.seg_files[self.shuffled_vector[files_index]]

            # judge the path
            if not os.path.exists(img_path):
                print ("{} not exists.".format(img_path))

            if not os.path.exists(seg_path):
                print ("{} not exists.".format(seg_path))

            img = cv2.imread(img_path)  # BGR
            seg = Image.open(seg_path)

            if img is None:
                continue

            self.augmentImageHSV(img)

            h, w, _ = img.shape
            img, ratio, padw, padh = self.trainDataProcess.resize_square(img,
                                                                         (width, height),
                                                                         self.color)
            #----------seg------------------------------------------------------------------
            ratio = min(float(width) / w, float(height) / h)  # ratio  = old / new
            new_shape = [round(h * ratio), round(w * ratio)]
            seg = self.trainDataProcess.encode_segmap(np.array(seg, dtype=np.uint8), self.volid_label_seg, self.valid_label_seg)
            seg = cv2.resize(seg, (new_shape[1], new_shape[0]))
            seg = np.pad(seg, ((padh // 2, padh - (padh // 2)), (padw // 2, padw - (padw // 2))), 'constant', constant_values=250)
            ################################################################################

            # Augment image and labels
            if self.augment:
                img, seg, M = self.random_affine(img, Segment=seg, degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.1))

            if self.augment:
                # random left-right flip
                lr_flip = True
                if lr_flip & (random.random() > 0.5):
                    img = np.fliplr(img)
                    seg = np.fliplr(seg)

            # seg
            valid_masks = np.zeros(seg.shape)
            for l in range(0, len(self.valid_label_seg)):
                valid_mask = seg == l # set false to position of seg that not in valid_label_seg
                valid_masks += valid_mask # set 0.0 to position of seg that not in valid_label_seg
            valid_masks[valid_masks==0]=-1
            seg = np.float32(seg) * valid_masks
            seg[seg < 0] = 250
            seg = np.uint8(seg)

            img_all.append(img)
            seg_all.append(seg)

        numpyImages = np.stack(img_all)[:, :, :, ::-1]
        torchImages = self.convertTorchTensor(numpyImages)

        seg_all = torch.from_numpy(np.array(seg_all)).long()

        return torchImages, seg_all

    def augmentImageHSV(self, inputRGBImage):
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(inputRGBImage, cv2.COLOR_RGB2HSV)
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
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=inputRGBImage)

    def __len__(self):
        return self.nB  # number of batches