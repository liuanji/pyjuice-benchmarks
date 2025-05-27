import numpy as np
import torch
import os
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import v2

class MNIST(Dataset):
    def __init__(self, root_dir = "/scratch/anji/data/MNIST", train = True, transform_fns = None):
        self.train = train

        self.dataset = torchvision.datasets.MNIST(root_dir, train, download = True)

        if transform_fns is not None:
            transforms = []
            for transform_fn in transform_fns:
                transforms.append(instantiate_from_config(transform_fn))
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = lambda x: x

    def __getitem__(self, index):
        img = self.dataset[index]

        return self.transforms(img)

    def __len__(self):
        return len(self.dataset)
