from .dataLoader import *

class ImageSegmentTrainDataLoader(DataLoader):
    def __init__(self, path, batch_size=1, img_size=[768, 320], augment=False):
        super().__init__(path)
        # self.img_files = sorted(glob.glob('%s/*.*' % path))
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        if sys.platform == 'darwin':  # MacOS (local)
            self.img_files = [path.replace('\n', '').replace('/images', './data/coco/images')
                              for path in self.img_files]
        else:  # linux (gcp cloud)
            self.img_files = [path.replace('\n', '').replace('/images', './data/coco/images') for path in self.img_files]

        self.seg_files = [path.replace('JPEGImages', 'SegmentLabel').replace('.png', '.png').replace('.jpg', '.png') for path in
                            self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]
        # self.volid_label_seg = [0, 1, 4, 5, 8, 9, 10, 11, 12, 14, 16, 18, 22, 25, 26, 28, 31, 32, 33, 34, 35, 36,
        #                         37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 53, 58, 59, 60, 62, 63, 64, 65]
        # self.valid_label_seg = [[54, 55, 56, 61], [19, 20, 21], [52, 57], [7, 13], [2, 3, 15, 29], [6, 17], [30],
        #                         [27], [23, 24], [48], [49, 50]]
        self.volid_label_seg = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_label_seg = [[7], [8], [11], [12], [13], [17], [19], [20], [21], [22],
                                [23], [24], [25], [26], [27], [28], [31], [32], [33]]

        self.augment = augment

        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

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

        multi_scale = False
        if multi_scale and self.augment:
            # Multi-Scale YOLO Training
            height = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            width = int(float(self.width) / float(self.height) * height)
        else:
            # Fixed-Scale YOLO Training
            height = self.height
            width = self.width

        img_all = []
        seg_all = []
        for index, files_index in enumerate(range(ia, ib)):
            img_path = self.img_files[self.shuffled_vector[files_index]]
            seg_path = self.seg_files[self.shuffled_vector[files_index]]

            img = cv2.imread(img_path)  # BGR
            seg = Image.open(seg_path)
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
            #----------seg------------------------------------------------------------------
            ratio = min(float(width) / w, float(height) / h)  # ratio  = old / new
            new_shape = [round(h * ratio), round(w * ratio)]
            seg = self.encode_segmap(np.array(seg, dtype=np.uint8), self.volid_label_seg, self.valid_label_seg)
            #seg = np.array(seg, dtype=np.uint8)
            #print(set(list(seg.flatten())))
            seg = cv2.resize(seg, (new_shape[1], new_shape[0]))
            seg = np.pad(seg, ((padh // 2, padh - (padh // 2)), (padw // 2, padw - (padw // 2))), 'constant', constant_values=250)
            #seg[(seg != 0) & (seg != 250)] = 1
            #seg = decode_segmap(seg)
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
                valid_mask = seg == l
                valid_masks += valid_mask
            valid_masks[valid_masks==0]=-1
            seg = np.float32(seg) * valid_masks
            seg[seg < 0] = 250
            seg = np.uint8(seg)
            # print(set(list(seg.flatten())))
            # segdraw = decode_segmap(seg)
            # cv2.imshow("Seg", segdraw)
            # cv2.waitKey()

            img_all.append(img)
            seg_all.append(seg)

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        # img_all -= self.rgb_mean
        # img_all /= self.rgb_std
        img_all /= 255.0

        #-------------seg
        seg_all = torch.from_numpy(np.array(seg_all)).long()

        return torch.from_numpy(img_all), seg_all

    def __len__(self):
        return self.nB  # number of batches