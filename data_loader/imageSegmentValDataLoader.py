from .dataLoader import *
from .trainDataProcess import TrainDataProcess
from PIL import Image

class ImageSegmentValDataLoader(DataLoader):
    def __init__(self, path, batch_size=1, img_size=(768, 320)):
        super().__init__(path)
        # self.img_files = sorted(glob.glob('%s/*.*' % path))
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
        # self.volid_label_seg = [0, 1, 4, 5, 8, 9, 10, 11, 12, 14, 16, 18, 22, 25, 26, 28, 31, 32, 33, 34, 35, 36,
        #                         37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 53, 58, 59, 60, 62, 63, 64, 65]
        # self.valid_label_seg = [[54, 55, 56, 61], [19, 20, 21], [52, 57], [7, 13], [2, 3, 15, 29], [6, 17], [30],
        #                         [27], [23, 24], [48], [49, 50]]
        self.volid_label_seg = []
        self.valid_label_seg = [[0], [1]]

        assert self.nB > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.arange(self.nF)
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

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

            h, w, _ = img.shape
            img, ratio, padw, padh = self.trainDataProcess.resize_square(img, (width, height), self.color)
            #----------seg------------------------------------------------------------------
            ratio = min(float(width) / w, float(height) / h)  # ratio  = old / new
            new_shape = [round(h * ratio), round(w * ratio)]
            seg = self.trainDataProcess.encode_segmap(np.array(seg, dtype=np.uint8), self.volid_label_seg, self.valid_label_seg)
            #seg = np.array(seg, dtype=np.uint8)
            #print(set(list(seg.flatten())))
            seg = cv2.resize(seg, (new_shape[1], new_shape[0]))
            seg = np.pad(seg, ((padh // 2, padh - (padh // 2)), (padw // 2, padw - (padw // 2))), 'constant', constant_values=250)
            ################################################################################

            # seg
            valid_masks = np.zeros(seg.shape)
            for l in range(0, len(self.valid_label_seg)):
                valid_mask = seg == l
                valid_masks += valid_mask
            valid_masks[valid_masks==0]=-1
            seg = np.float32(seg) * valid_masks
            seg[seg < 0] = 250
            seg = np.uint8(seg)

            img_all.append(img)
            seg_all.append(seg)

        numpyImages = np.stack(img_all)[:, :, :, ::-1]
        torchImages = self.convertTorchTensor(numpyImages)

        #-------------seg
        seg_all = torch.from_numpy(np.array(seg_all)).long()

        return torchImages, seg_all

    def __len__(self):
        return self.nB  # number of batches