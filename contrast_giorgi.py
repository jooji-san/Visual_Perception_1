import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Grayscale, GaussianBlur
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import time
import random

def main():
    root = './data'

    #Loads the CIFAR-10 dataset
    cifar_dataset = CIFAR10(root=root, train=False, download=True,
                            transform=Compose(
                                [ToTensor(),
                                 Grayscale(num_output_channels=1)]))

    ages = [0, 6, 12]
    batch_size = 1

    #Creating loaders
    for age in ages:
        print(f"Visualizing transformations for age {age} months")
        loaders = {
            f"With Visual Acuity (Age {age})": DataLoader(
                InfantVisionDataset(cifar_dataset, age_in_months=age, apply_acuity=True, apply_contrast=False),
                batch_size=batch_size, shuffle=False
            ),
            f"With Spatial Frequency Contrast (Age {age})": DataLoader(
                InfantVisionDataset(cifar_dataset, age_in_months=age, apply_acuity=False, apply_contrast=True),
                batch_size=batch_size, shuffle=False
            ),
            f"With Both Transformations (Age {age})": DataLoader(
                InfantVisionDataset(cifar_dataset, age_in_months=age, apply_acuity=True, apply_contrast=True),
                batch_size=batch_size, shuffle=False
            ),
        }

        #Visualize and evaluate performance for each loader
        for name, loader in loaders.items():
            print(f"Visualizing {name}")
            visualize_transforms(loader, num_images=5)
            time_taken = evaluate_performance(loader, name)
            print(f"Time taken: {time_taken:.2f} seconds")

class InfantVisionDataset(Dataset):
    def __init__(self, dataset, age_in_months=0, apply_acuity=False, apply_contrast=False):
        self.dataset = dataset
        self.age_in_months = age_in_months
        self.apply_acuity = apply_acuity
        self.apply_contrast = apply_contrast

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        processed_image = image

        if self.apply_acuity:
            processed_image = apply_visual_acuity(image, self.age_in_months)

        if self.apply_contrast:
            processed_image = apply_spatial_frequency_contrast(image, self.age_in_months)

        return image, processed_image, label

############################################################################################################
# Visual Acuity

def map_visual_acuity(age_months):
    """Map age in months to blur kernel size based on visual acuity (20/600 to 20/20)."""
    min_acuity = 600  #20/600 for newborns
    max_acuity = 20   #20/20 for 12 months onwards
    acuity = min_acuity - (min_acuity - max_acuity) * (age_months / 12)
    kernel_size = max(3, int((acuity / min_acuity) * 11))  #Scales the kernel size, min 3
    return kernel_size

def apply_visual_acuity(image, age_months):
    """Apply visual acuity (blurring) based on age in months."""
    kernel_size = map_visual_acuity(age_months)
    blurred_image = GaussianBlur(kernel_size)(image)
    return blurred_image

############################################################################################################
# Contrast

def create_frequency_masks(size):
    """Create frequency band masks using PyTorch."""
    rows, cols = size
    center_row, center_col = rows // 2, cols // 2

    y_grid, x_grid = torch.meshgrid(
        torch.arange(rows, dtype=torch.float32),
        torch.arange(cols, dtype=torch.float32),
        indexing='ij'
    )

    #Formula for distance from center
    distance = torch.sqrt(
        (y_grid - center_row) ** 2 + (x_grid - center_col) ** 2
    ) / (min(rows, cols) / 2)

    #Frequency bands
    low_mask = (distance <= 0.2).float()
    mid_mask = ((distance > 0.2) & (distance <= 0.5)).float()
    high_mask = (distance > 0.5).float()

    return [low_mask, mid_mask, high_mask]

def separate_frequencies(image):
    """Separate image into different frequency bands using FFT."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    fft = torch.fft.fft2(image.float())
    fft_shift = torch.fft.fftshift(fft)

    masks = [mask.to(device) for mask in create_frequency_masks(image.shape[-2:])]
    bands = []
    for mask in masks:
        filtered = fft_shift * mask
        filtered_shift = torch.fft.ifftshift(filtered)
        reconstructed = torch.fft.ifft2(filtered_shift).real
        bands.append(reconstructed)

    return bands

def adjust_contrast(image, factor):
    """Adjust contrast of an image using PyTorch"""
    mean = torch.mean(image)
    return (image - mean) * factor + mean

def get_sensitivity_factors(age_months):
    """Get sensitivity factors for different frequency bands."""
    return [1, 1, 1]

def apply_spatial_frequency_contrast(image, age_months):
    sensitivity_factors = get_sensitivity_factors(age_months)

    """Apply contrast sensitivity to different frequency bands."""
    frequency_bands = separate_frequencies(image)
    transformed_bands = [
        adjust_contrast(band.unsqueeze(0), factor).squeeze(0)
        for band, factor in zip(frequency_bands, sensitivity_factors)
    ]
    return torch.sum(torch.stack(transformed_bands), dim=0)

############################################################################################################

def evaluate_performance(loader, name="Dataset"):
    """Measure the performance of data loading."""
    start_time = time.time()
    for i, _ in enumerate(loader):
        if i >= 1:  #Only load 100 images (1 batch of 100 images)
            break
    elapsed_time = time.time() - start_time
    print(f"Time to load 100 images from {name}: {elapsed_time:.2f} seconds")
    return elapsed_time

def visualize_transforms(loader, num_images=5):
    """Visualize original and transformed images."""
    #Randomly select indices
    indices = random.sample(range(len(loader.dataset)), num_images)
    
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(indices):
        original, transformed, _ = loader.dataset[idx]
        
        #Displays original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original.permute(1, 2, 0), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        #Displays transformed image
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(transformed.permute(1, 2, 0), cmap='gray')
        plt.title("Transformed")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()



