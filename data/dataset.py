from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import time


# pixel art dataset loader.
class PixelArtDataset(Dataset):
    def __init__(self, base_path, transform=None, gauge_collection=None):
        self.labels = np.load(f"{base_path}/sprites_labels.npy")
        self.base_path = base_path
        self.num_images = self.labels.shape[0]
        self.transform = transform
        self.gauge_collection = gauge_collection

        # Create gauges if collection is provided
        if self.gauge_collection:
            self.image_load_gauge = self.gauge_collection.create_gauge("image_load")
            self.preprocess_gauge = self.gauge_collection.create_gauge("preprocess")
        else:
            self.image_load_gauge = None
            self.preprocess_gauge = None

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        label = self.labels[idx]

        # Measure image loading time
        if self.image_load_gauge:
            start_time = time.time()

        img_path = f"{self.base_path}/images/images/image_{idx}.JPEG"
        image = Image.open(img_path)

        if self.image_load_gauge:
            load_duration = time.time() - start_time
            # Estimate size from image dimensions (width * height * channels)
            image_size = image.width * image.height * len(image.getbands())
            self.image_load_gauge.record_event(load_duration, image_size)

        label = "".join([str(item) for item in torch.from_numpy(label).float()])

        batch = {
            "image": [image],
            "label": [label],
        }

        if self.transform is None:
            raise ValueError("Transform is not set")

        # Measure preprocessing time
        if self.preprocess_gauge:
            start_time = time.time()

        transformed = self.transform(batch)

        if self.preprocess_gauge:
            preprocess_duration = time.time() - start_time
            # Use same size estimate for preprocessing
            self.preprocess_gauge.record_event(preprocess_duration, image_size)

        return {key: value[0] for key, value in transformed.items()}

    def with_transform(self, transform):
        self.transform = transform
        return self

    @property
    def column_names(self):
        return ["image", "label"]


def load_dataset(base_path, gauge_collection=None):
    return {"train": PixelArtDataset(base_path, gauge_collection=gauge_collection)}
