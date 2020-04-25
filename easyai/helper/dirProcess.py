import os
import os.path
import glob

class DirProcess():

    def __init__(self):
        pass

    def getDirFiles(self, dataDir, filePost="*.*"):
        if os.path.isdir(dataDir):
            imagePathPattern = os.path.join(dataDir, filePost)
            for filePath in glob.iglob(imagePathPattern):
                yield filePath
            return
        else:
            return None

    def getFileData(self, dataFilePath):
        with open(dataFilePath, 'r') as file:
            for line in file:
                if line.strip():
                    yield line.strip()
        return