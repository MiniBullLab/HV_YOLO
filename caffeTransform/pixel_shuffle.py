import torch
import numpy as np

class upsample():
    def __init__(self, scaleFactors):
        self.scale = scaleFactors

    def reorg(self, input):
        b, c, h, w = input.shape

        out_c = c / (self.scale * self.scale)
        output = np.zeros([b, int(out_c), int(h*self.scale), int(w*self.scale)])

        for bn in range(0, b):
            for k in range(0, c):
                for j in range(0, h):
                    for i in range(0, w):
                        c2 = int(k % out_c)
                        offset = k / out_c
                        h2 = int(j * self.scale + offset / self.scale)
                        w2 = int(i * self.scale + offset % self.scale)
                        output[bn, c2, h2, w2] = input[bn, k, j, i]

        return output

    def pixelS(self, input):
        b, c, h, w = input.shape

        out_c = c / (self.scale * self.scale)
        rSquare = self.scale * self.scale
        output = np.zeros([b, int(out_c), int(h*self.scale), int(w*self.scale)])

        for bn in range(0, b):
            for k in range(0, c):
                for j in range(0, h):
                    for i in range(0, w):
                        c2 = int(k / rSquare)
                        h2 = int(j * self.scale + ((k / self.scale) % self.scale))
                        w2 = int(i * self.scale + k % self.scale)
                        output[bn, c2, h2, w2] = input[bn, k, j, i]

        return output

if __name__ == "__main__":
    up = upsample(2)

    shape = (1, 16, 4, 4)
    input = torch.randn(shape)
    x = input.numpy()
    upS = torch.nn.PixelShuffle(2)
    out = up.reorg(x)
    outPSh = up.pixelS(x)
    outUPS = upS(input).numpy()
    print("out: {}".format(out))
    print("outPSh: {}".format(outPSh))
    print("outUPS: {}".format(outUPS))