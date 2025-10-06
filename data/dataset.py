from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import time


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

        # Measure image loading time
        load_start = time.time()
        img_path = f"{self.base_path}/images/images/image_{idx}.JPEG"
        image = Image.open(img_path)
        load_duration = time.time() - load_start

        # Estimate size from image dimensions (width * height * channels)
        image_size = image.width * image.height * len(image.getbands())

        label = "".join([str(item) for item in torch.from_numpy(label).float()])

        batch = {
            "image": [image],
            "label": [label],
        }

        if self.transform is None:
            raise ValueError("Transform is not set")

        # Measure preprocessing time
        preprocess_start = time.time()
        transformed = self.transform(batch)
        preprocess_duration = time.time() - preprocess_start

        result = {key: value[0] for key, value in transformed.items()}

        # Add timing metadata to the result
        result["_timing_image_load"] = load_duration
        result["_timing_image_size"] = image_size
        result["_timing_preprocess"] = preprocess_duration

        return result

    def with_transform(self, transform):
        self.transform = transform
        return self

    @property
    def column_names(self):
        return ["image", "label"]


def load_dataset(base_path):
    return {"train": PixelArtDataset(base_path)}
