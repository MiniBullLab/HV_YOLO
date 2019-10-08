import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from base_model.baseModel import *
from modelName import ModelName
import torch.nn.init as init


class MySRModel(BaseModel):
    def __init__(self, upscale_factor):
        super().__init__()
        self.setModelName(ModelName.MySRModel)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


if __name__ == "__main__":
    from drawing.modelNetShow import ModelNetShow
    input = torch.randn(1, 1, 72, 72)
    modelNetShow = ModelNetShow()
    model = MySRModel(upscale_factor=3)
    modelNetShow.setInput(input)
    modelNetShow.setSaveDir("../onnx")
    modelNetShow.showNet(model)