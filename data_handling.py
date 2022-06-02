import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import random_split

def change_to_numeric_binary(boolean_string):
    """
    Converts a 'yes' or 'no' to a 1 or 0
    """
    value = 0
    if boolean_string == 'yes':
                value = 1
    return value

class NeedleImageDataset(Dataset):
    def __init__(self, path2data, path2labels, transform, data_type='train'):
        """
        Initializer
        """
        self.path2data = path2data
        self.path2labels = path2labels
        labels_df = pd.read_csv(path2labels)
        labels_df = labels_df[labels_df['Filename'] != '20410.jpg']
        assert ('Label' in labels_df.keys()) and ('Filename' in labels_df.keys())
        self.example_label_map = dict(zip(labels_df.Filename, labels_df.Label))
        self.filenames = list(self.example_label_map.keys())
        self.transform = transform
        
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.example_label_map)
        
    def join(self, filename):
        """
        Helper for getting the full path to an image
        """
        return os.path.join(self.path2data, filename)
        
    def __getitem__(self, idx):
        """
        Returns a single example and corresponding label
        """
        filename = self.filenames[idx]
        image = Image.open(self.join(filename)).convert('RGB')
        image = self.transform(image)
        return image, change_to_numeric_binary(self.example_label_map[filename])
