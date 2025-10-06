from torch.utils.data import Dataset
import numpy as np
import torch


# pixel art dataset loader.
class PixelArtDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.images = np.load(f"{base_path}/sprites.npy")
        self.labels = np.load(f"{base_path}/sprites_labels.npy")
        self.transform = transform

        # Data Validation (more comprehensive)
        if self.images.shape[1:] != (16, 16, 3):  # Check image size (16x16x3)
            raise ValueError(f"Images must be 16x16x3, but are {self.images.shape}")
        if self.labels.ndim != 2 or self.labels.shape[1] != 5:  # Check label shape
            raise ValueError(
                f"Labels must be a 2D array with 5 elements per row, but are {self.labels.shape}"
            )
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("Number of images and labels must match")
        if (
            self.images.dtype != np.uint8
        ):  # Check data type. Assuming your images are 8-bit
            raise TypeError(
                f"Images must be uint8, but are {self.images.dtype}. Convert if necessary"
            )
        if (
            self.labels.dtype != np.float64
        ):  # Check data type. Assuming your labels are float32
            raise TypeError(
                f"Labels must be float32, but are {self.labels.dtype}. Convert if necessary"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image to tensor, normalize, and add channel dimension
        image = torch.from_numpy(image).float() / 255.0  # Normalize to [0, 1]

        # No need to unsqueeze for RGB images:
        image = image.permute(2, 0, 1)  # Change from HxWx3 to 3xHxW (CHW)

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(label).float()

        return image, label

    @property
    def column_names(self):
        return ["image", "label"]


def load_dataset(base_path):
    return {"train": PixelArtDataset(base_path)}
