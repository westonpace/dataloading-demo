import argparse
from pathlib import Path
import time
import torch
import torchvision
from torch.profiler import profile, ProfilerActivity
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from diffusers import UNet2DModel
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Data loading demo for pixel art dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/pace/dev/data/pixel-art",
        help="Base path to the data directory",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler",
    )
    parser.add_argument(
        "--print-profile",
        action="store_true",
        help="Print profiler key averages table",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([])

    dataset = PixelArtDataset(data_path, transform=transform)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=True
    )  # Adjust batch size as needed

    x, y = next(iter(dataloader))
    print("Input shape:", x.shape)  # Should be (batch_size, channels, height, width)
    # print('Labels:', y)

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # CHW to HWC
        plt.show()

    # Display the grid of images (all images in the batch)
    imshow(torchvision.utils.make_grid(x))

    def corrupt(x, amount):
        """Corrupt the input `x` by mixing it with noise according to `amount`"""
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
        return x * (1 - amount) + noise * amount

    # Create the network
    net = UNet2DModel(
        sample_size=16,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )
    net.to(device)

    n_epochs = args.num_epochs
    batch_size = 512
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    epoch_durations = []

    def train_loop():
        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            for x, y in train_dataloader:
                # Get some data and prepare the corrupted version
                x = x.to(device)  # Data on the GPU
                noise_amount = torch.rand(x.shape[0]).to(
                    device
                )  # Pick random noise amounts
                noisy_x = corrupt(x, noise_amount)  # Create our noisy x

                # Get the model prediction
                pred = net(
                    noisy_x, 0
                ).sample  # <<< Using timestep 0 always, adding .sample

                # Calculate the loss
                loss = loss_fn(
                    pred, x
                )  # How close is the output to the true 'clean' x?

                # Backprop and update the params:
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Store the loss for later
                losses.append(loss.item())

            epoch_duration = time.time() - epoch_start_time
            epoch_durations.append(epoch_duration)

            # Print our the average of the loss values for this epoch:
            avg_loss = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(
                f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}, Duration: {epoch_duration:.2f}s"
            )

    if args.profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            train_loop()

        if args.print_profile:
            # Analyze the results
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # Export for detailed analysis
        prof.export_chrome_trace("trace.json")
    else:
        train_loop()

    total_duration = sum(epoch_durations)
    avg_duration = total_duration / len(epoch_durations)
    print(
        f"\nTotal duration: {total_duration:.2f}s, Average duration per epoch: {avg_duration:.2f}s"
    )


# pixel art dataset loader.
class PixelArtDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
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


if __name__ == "__main__":
    main()
