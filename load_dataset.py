import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile as tiff
import numpy as np
from sklearn.model_selection import train_test_split

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, split, bands, val_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.split = split
        self.bands = [int(band) for band in bands.split(",")]
        self.val_size = val_size
        self.random_state = random_state
        self.transform = self._get_transform()
        self.image_files, self.labels = self._load_data()

    def _load_data(self):
        image_files = []
        labels = []

        label_file = os.path.join(self.data_dir, "train/answer.csv")
        with open(label_file, "r") as f:
            for line in f:
                image_file, label = line.strip().split(",")
                image_files.append(os.path.join(self.data_dir, "train/train", image_file))
                labels.append(int(label))

        # Split the data into training and validation sets
        train_files, val_files, train_labels, val_labels = train_test_split(
            image_files, labels, test_size=self.val_size, stratify=labels, random_state=self.random_state
        )

        if self.split == "train":
            return train_files, train_labels
        elif self.split == "val":
            return val_files, val_labels
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _get_transform(self):
        if self.split == "train":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])
        else:
            transform = transforms.Compose([])
        return transform

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        image = tiff.imread(image_file)
        assert image.shape == (512, 512, 12), f"Expected shape (512, 512, 12), but got {image.shape}"

        selected_bands = image[:, :, self.bands]  # Select the desired bands
        selected_bands = selected_bands.astype(np.float32) / 65535.0  # Normalize to [0, 1]
        selected_bands = np.transpose(selected_bands, (2, 0, 1))  # Change to (C, H, W) format
        selected_bands = torch.from_numpy(selected_bands)  # Convert to PyTorch tensor

        if self.transform:
            selected_bands = self.transform(selected_bands)

        selected_bands = selected_bands.to(torch.float32)  # Convert to the desired data type

        return selected_bands, label

    def __len__(self):
        return len(self.image_files)