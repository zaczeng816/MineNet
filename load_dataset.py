import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tiff
import numpy as np

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, split, bands):
        self.data_dir = data_dir
        self.split = split
        self.bands = [int(band) for band in bands.split(",")]
        self.transform = self._get_transform()
        self.image_files, self.labels = self._load_data()

    def _load_data(self):
        image_files = []
        labels = []

        if self.split == "train":
            label_file = os.path.join(self.data_dir, "answer.csv")
            with open(label_file, "r") as f:
                for line in f:
                    image_file, label = line.strip().split(",")
                    image_files.append(os.path.join(self.data_dir, "train", image_file))
                    labels.append(int(label))
        else:
            # Implement loading of validation/test data if applicable
            pass

        return image_files, labels

    def _get_transform(self):
        if self.split == "train":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        image = tiff.imread(image_file)
        selected_bands = image[:, :, self.bands]
        selected_bands = selected_bands.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        selected_bands = np.transpose(selected_bands, (2, 0, 1))  # Change to (C, H, W) format

        if self.transform:
            selected_bands = self.transform(selected_bands)

        return selected_bands, label

    def __len__(self):
        return len(self.image_files)