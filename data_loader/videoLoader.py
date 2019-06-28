from .dataLoader import *

class VideoLoader(DataLoader):
    '''
    load video
    '''
    def __init__(self, path, batch_size=1, img_size=[416, 416]):
        super().__init__(path)
        if not os.path.exists(path):
            raise Exception("Not exists path!", path)
        if not path.endswith(".MPG") and not path.endswith(".avi") and not path.endswith(".mp4"):
            raise Exception("Invalid path!", path)

        self.videoCapture = cv2.VideoCapture(path)
        self.batch_size = batch_size
        self.width = img_size[0]
        self.height = img_size[1]

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        img_all = []
        frame_all = []

        for i in range(0, self.batch_size):
            success, frame = self.videoCapture.read()

            if success == False:
                break

            # Padded resize
            img, _, _, _ = self.resize_square(frame, width=self.width, height=self.height, color=(127.5, 127.5, 127.5))
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_all.append(rgbImage)
            frame_all.append(frame)

        if len(img_all) > 0:
            img_all = np.ascontiguousarray(img_all, dtype=np.float32)
            # img_all -= self.rgb_mean
            # img_all /= self.rgb_std
            img_all /= 255.0

            return frame_all, torch.from_numpy(img_all)
        else:
            raise StopIteration