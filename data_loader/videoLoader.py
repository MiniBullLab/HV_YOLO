from helper import VideoProcess
from data_loader.dataLoader import *
from data_loader.trainDataProcess import TrainDataProcess


class VideoLoader(DataLoader):
    """
    load video
    """
    def __init__(self, path, batch_size=1, img_size=(416, 416)):
        super().__init__(path)
        self.videoProcess = VideoProcess()
        self.trainDataProcess = TrainDataProcess()
        if not self.videoProcess.isVideoFile(path) or \
                not self.videoProcess.openVideo(path):
            raise Exception("Invalid path!", path)

        self.color = (127.5, 127.5, 127.5)
        self.batch_size = batch_size
        self.imageSize = img_size
        self.frameCount = self.videoProcess.getFrameCount()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        success, srcImage, rgbImage = self.videoProcess.readRGBFrame()

        if success == False:
            raise StopIteration
        # Padded resize
        rgbImage, _, _, _ = self.trainDataProcess.resize_square(rgbImage, self.imageSize, self.color)
        torchImage = self.convertTorchTensor(rgbImage)

        return srcImage, torchImage

    def __len__(self):
        return int(self.frameCount)  # number of batches