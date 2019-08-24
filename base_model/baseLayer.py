import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super().__init__()

class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]

        if torch.is_tensor(h) or torch.is_tensor(w):
            h = np.asarray(h)
            w = np.asarray(w)
            return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))
        else:
            return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))

class FcLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    pass
