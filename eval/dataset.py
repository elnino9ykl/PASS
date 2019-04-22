# Code with dataset loader
# April 2019
# Kailun Yang
#######################

import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png', '.JPG']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)
    #return filename.endswith("color.png")

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, subset='val'):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
       

    def __getitem__(self, index):
        filename = self.filenames[index]
       
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')

        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, filename

    def __len__(self):
        return len(self.filenames)

