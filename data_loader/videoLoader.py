from .dataLoader import *
from .trainDataProcess import TrainDataProcess

class VideoLoader(DataLoader):
    '''
    load video
    '''
    def __init__(self, path, batch_size=1, img_size=(416, 416)):
        super().__init__(path)
        if not os.path.exists(path):
            raise Exception("Not exists path!", path)
        if not path.endswith(".MPG") and not path.endswith(".avi") and not path.endswith(".mp4"):
            raise Exception("Invalid path!", path)

        self.trainDataProcess = TrainDataProcess()
        self.videoCapture = cv2.VideoCapture(path)
        self.batch_size = batch_size
        self.imageSize = img_size
        self.color = (127.5, 127.5, 127.5)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        for i in range(0, self.batch_size):
            success, frame = self.videoCapture.read()

            if success == False:
                break
            oriImg = frame[:]
            # Padded resize
            img, _, _, _ = self.trainDataProcess.resize_square(frame, self.imageSize, self.color)
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if frame:
            rgbImage = rgbImage.transpose(2, 0, 1)
            img = np.ascontiguousarray(rgbImage, dtype=np.float32)
            img /= 255.0

            return oriImg, torch.from_numpy(img).unsqueeze(0)
        else:
            raise StopIteration