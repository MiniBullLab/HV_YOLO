import torch

class TensorsDataset(torch.utils.data.Dataset):

    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, transforms=None):
        self.data_tensor = data_tensor

        if transforms is None:
            transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]

        self.transforms = transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        return data_tensor

    def __len__(self):
        return self.data_tensor.size(0)