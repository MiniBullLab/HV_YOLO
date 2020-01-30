import os
import cv2


class VideoProcess():

    def __init__(self):
        self.videoPath = ""
        self.videoCapture = None

    def isVideoFile(self, videoPath):
        if os.path.exists(videoPath):
            return any(videoPath.endswith(extension) for extension in [".avi", ".mp4", ".mov", ".MPG"])
        else:
            return False

    def openVideo(self, videoPath):
        self.videoCapture = cv2.VideoCapture(videoPath)
        if self.videoCapture is not None:
            return True
        else:
            return False

    def getFrameCount(self):
        result = 0
        if self.videoCapture is not None:
            result = self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        return result

    def readFrame(self):
        success = False
        frame = None
        if self.videoCapture is not None:
            success, frame = self.videoCapture.read()
        return success, frame

    def readRGBFrame(self):
        success = False
        frame = None
        rgbImage = None
        if self.videoCapture is not None:
            success, frame = self.videoCapture.read()
        if success == True:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return success, frame, rgbImage