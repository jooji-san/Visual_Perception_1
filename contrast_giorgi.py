from math import tan, pi
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Grayscale, GaussianBlur
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from scipy.interpolate import interp1d

def main():
    root = './data'

    #Loads the CIFAR-10 dataset
    cifar_dataset = CIFAR10(root=root, train=False, download=True,
                            transform=Compose(
                                [ToTensor(),
                                 Grayscale(num_output_channels=1)]))

    ages = [1, 2, 3]
    batch_size = 100

    #Store performance results
    performance_results = []

    #Add DataLoader for no transformations
    no_transform_loader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=False)
    time_taken = evaluate_performance(no_transform_loader, "Without Transformations")
    performance_results.append({"Transformation": "Without Transformations", "Time (s)": time_taken})

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
            )
        }

        #Visualize and evaluate performance for each loader
        for name, loader in loaders.items():
            visualize_transforms(f"Visualizing {name}", loader, num_images=5)
            time_taken = evaluate_performance(loader, name)
            performance_results.append({"Transformation": name, "Time (s)": time_taken})
            print(f"Time taken: {time_taken:.2f} seconds")

    #Create the dot plot
    plot_performance(performance_results)

class InfantVisionDataset(Dataset):
    def __init__(self, dataset, age_in_months=0, apply_acuity=False, apply_contrast=False):
        self.dataset = dataset
        self.age_in_months = age_in_months
        self.apply_acuity = apply_acuity
        self.apply_contrast = apply_contrast
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        processed_image = image

        if self.apply_acuity:
            processed_image = apply_visual_acuity(image, self.age_in_months)

        if self.apply_contrast:
            processed_image = apply_spatial_frequency_contrast(self.device, image, self.age_in_months)

        return image, processed_image, label

############################################################################################################
#Visual Acuity

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
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred_image = GaussianBlur(kernel_size)(image)
    return blurred_image

############################################################################################################
#Contrast

def cpd_to_normalized_freq(cpd, image_width_pixels, viewing_distance_mm, screen_width_mm):
    """Convert cycles per degree to normalized frequency."""
    #Calculate pixels per degree
    ppd = (viewing_distance_mm * tan(1 * pi/180) * image_width_pixels) / screen_width_mm
    #Convert cycles per degree to cycles per pixel
    cpp = cpd / ppd
    #Convert to normalized frequency (0 to 1)
    return cpp * image_width_pixels


def create_vertical_frequency_masks(size, frequency_bounds):
    """Create vertical frequency band masks using PyTorch."""
    rows, cols = size
    center_row, center_col = rows // 2, cols // 2

    y_grid, x_grid = torch.meshgrid(
        torch.arange(rows, dtype=torch.float32),
        torch.arange(cols, dtype=torch.float32),
        indexing='ij'
    )

    vertical_freq = torch.abs(x_grid - center_col) / (cols / 2)

    #Convert frequency bounds to tensor if it's not already
    freq_bounds = torch.tensor(frequency_bounds, dtype=torch.float32)

    masks = []
    for i in range(len(freq_bounds) - 1):
        mask = ((vertical_freq > freq_bounds[i]) &
                (vertical_freq <= freq_bounds[i + 1])).float()
        masks.append(mask)

    return masks

def separate_vertical_frequencies(device, image, frequency_bounds):
    """Separate image into different frequency bands using FFT."""
    image = image.to(device)

    fft = torch.fft.fft2(image.float())
    fft_shift = torch.fft.fftshift(fft)

    masks = [mask.to(device) for mask in create_vertical_frequency_masks(image.shape[-2:], frequency_bounds)]
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

def interpolate_sensitivity(known_sensitivities, known_frequencies, target_frequencies):
    known_freq = np.array(known_frequencies)
    known_sens = np.array(known_sensitivities)

    interpolator = interp1d(known_freq, known_sens,
                            kind='linear',
                            fill_value=(known_sens[0], known_sens[-1]),
                            bounds_error=False)

    return torch.from_numpy(interpolator(target_frequencies)).float()

def get_sensitivity_factors(image, age_months, center_frequencies):
    """
    Get sensitivity factors for different frequency bands.
    """
    known_frequencies_cpd = [0.2, 0.3, 0.5, 1, 2]
    max_cpd = 30.0
    known_frequencies_norm = [cpd/max_cpd for cpd in known_frequencies_cpd]

    if age_months == 1:
        known_sensitivities = [7, 7, 8, 3, 2]
    elif age_months == 2:
        known_sensitivities = [5, 11, 10, 3, 3]
    else:
        known_sensitivities = [7, 11, 17, 10, 5]

    adult_sensitivities = [120, 190, 350, 380, 800]

    relative_sensitivity = [s1/s2 for s1, s2 in zip(known_sensitivities, adult_sensitivities)]

    factors = interpolate_sensitivity(relative_sensitivity,
                                      known_frequencies_norm,
                                      center_frequencies)

    return factors

def apply_spatial_frequency_contrast(device, image, age_months):
    #Define frequency bounds (in normalized units)
    max_cpd = 30.0
    cpd_bounds = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]  #in CPD
    frequency_bounds = torch.tensor([cpd / max_cpd for cpd in cpd_bounds], device=device)

    #Calculate center frequency for each band
    center_frequencies = [(frequency_bounds[i] + frequency_bounds[i + 1]) / 2
                          for i in range(len(frequency_bounds) - 1)]

    #Get sensitivity factors
    sensitivity_factors = get_sensitivity_factors(image, age_months, center_frequencies)
    sensitivity_factors = sensitivity_factors.to(device)

    #Separate frequencies
    vertical_frequency_bands = separate_vertical_frequencies(device, image, frequency_bounds.cpu().tolist())

    vertical_transformed_bands = [
        adjust_contrast(band.unsqueeze(0), factor).squeeze(0)
        for band, factor in zip(vertical_frequency_bands, sensitivity_factors)
    ]
    return torch.sum(torch.stack(vertical_transformed_bands), dim=0)

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

def plot_performance(performance_results):
    """Plot the dot plot for performance times."""
    plt.figure(figsize=(10, 5))
    for result in performance_results:
        plt.scatter(result["Transformation"], result["Time (s)"], s=100, color="blue")
    plt.title("Performance Comparison")
    plt.xlabel("Transformation Scenario")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def visualize_transforms(name, loader, num_images=5):
    """Visualize original and transformed images."""
    #Randomly select indices

    indices = random.sample(range(len(loader.dataset)), num_images)
    
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(indices):
        plt.suptitle(name, fontsize=16)

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
