from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image


# pixel art dataset loader.
class PixelArtDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.labels = np.load(f"{base_path}/sprites_labels.npy")
        self.base_path = base_path
        self.num_images = self.labels.shape[0]
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        label = self.labels[idx]

        img_path = f"{self.base_path}/images/images/image_{idx}.JPEG"
        image = Image.open(img_path)

        label = torch.from_numpy(label).float()

        batch = {
            "image": [image],
            "label": [label],
        }

        if self.transform:
            batch = self.transform(batch)

        return {
            "image": batch["image"][0],
            "label": batch["label"][0],
        }

    def with_transform(self, transform):
        self.transform = transform
        return self

    @property
    def column_names(self):
        return ["image", "label"]


def load_dataset(base_path):
    return {"train": PixelArtDataset(base_path)}
