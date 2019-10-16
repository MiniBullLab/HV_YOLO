import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.loss_name = name

    def set_loss_name(self, name):
        self.loss_name = name

    def get_loss_name(self):
        return self.loss_name
