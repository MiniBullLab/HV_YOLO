from .dataLoader import *

class ImageClassifyTrainDataLoader(DataLoader):
    def __init__(self, path, batch_size=1, img_size=[416, 416]):
        super().__init__(path)
        with open(path, 'r') as file:
            self.img_files = file.readlines()

        self.img_files = [path.replace('\n', '') for path in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]

        assert self.nF > 0, 'No images found in path %s' % path

        # RGB normalization values
        # self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        # self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size

        img_all = []
        labels_all = []

        for i in range(ia, ia + self.batch_size):
            # Read image
            img_path = self.img_files[i]
            imagePath = img_path.split(" ")[0]
            label = int(img_path.split(" ")[1])
            img = cv2.imread(imagePath)  # BGR
            img = cv2.resize(img, (self.height, self.width))

            # Padded resize
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_all.append(rgbImage)
            labels_all.append(label)

        if len(img_all) > 0:
            img_all = np.stack(img_all).transpose(0, 3, 1, 2)
            img_all = np.ascontiguousarray(img_all, dtype=np.float32)
            img_all /= 255.0

        return torch.from_numpy(img_all), torch.from_numpy(np.asarray(labels_all))

    def __len__(self):
        return self.nB  # number of batches