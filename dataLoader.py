import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime

age_data_acuity = np.array([0, 1, 3, 6, 12, 24, 36, 48])  
acuity_data = np.array([600, 581, 231, 109, 66, 36, 25, 20]) # 0 month should not be 687 as shown in the data, but 600 as shown in the exercise description

age_data_contrast = np.array([1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 216])  
contrast_data = np.array([0.054092562, 0.079797917, 0.087848712, 0.197312866, 0.301349368, 0.548930981, 0.64538004, 0.635222933, 0.675745399, 0.772787744, 0.824422479, 0.938288057, 1], dtype=float)


# File to save timings with aggregated total times
filename = f"timing_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
def log_to_file(message):
    with open(filename, "a") as file:
        file.write(message + "\n")

def age_to_acuity_scale(months):
    return np.interp(months, age_data_acuity, acuity_data) / 20
    
def age_to_contrast_scale(months):
    return np.interp(months, age_data_contrast, contrast_data)
    

class VisualAcuityDataset(Dataset):
    def __init__(self, image_dir, age_months):
        self.image_dir = image_dir
        self.age_months = age_months
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.acuity_scale = age_to_acuity_scale(age_months)
        max_blur_radius = 4
        self.blur_radius = max_blur_radius * (self.acuity_scale - (20 / 20)) / ((687 / 20) - (20 / 20))
        log_to_file(f"age months: {self.age_months}")

        # Initialize a variable to accumulate the filter times
        self.filter_time = 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Timing for blur operation
        start_time = time.time()
        blurred_image = image.filter(ImageFilter.GaussianBlur(self.blur_radius))
        elapsed_time = time.time() - start_time
        
        self.filter_time += elapsed_time
        # log_to_file(f"Time for blur operation (1 run): {elapsed_time} seconds")

        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(blurred_image)).float() / 255.0
        return image_tensor, self.age_months
    
    def get_total_filter_time(self):
        return self.filter_time


class VisualContrastDataset(Dataset):
    def __init__(self, image_dir, age_months):
        self.image_dir = image_dir
        self.age_months = age_months
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.contrast_scale = float(age_to_contrast_scale(age_months))
        log_to_file(f"age months: {self.age_months}")
        
        # Initialize a variable to accumulate the filter times
        self.filter_time = 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Timing for contrast operation
        start_time = time.time()
        contrast_enhancer = ImageEnhance.Contrast(image)
        contrast_image = contrast_enhancer.enhance(self.contrast_scale)
        elapsed_time = time.time() - start_time

        self.filter_time += elapsed_time
        # log_to_file(f"Time for contrast operation (1 run): {elapsed_time} seconds")

        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(contrast_image)).float() / 255.0
        return image_tensor, self.age_months
    
    def get_total_filter_time(self):
        return self.filter_time

def test_performance_comparison(image_dir, batch_size):
    repetitions = 5  # average 5 iterations for each parameter

    # Define age ranges for acuity and contrast datasets
    x_months_acuity = np.linspace(0, 48, 25).astype(int)
    x_months_contrast = np.linspace(0, 220, 23).astype(int)

    acuity_results = []
    contrast_results = []

    for k in x_months_acuity:
        total_time = 0
        for _ in range(repetitions):
            start_time = time.time()

            for batch_id, (images, ages) in enumerate(DataLoader(VisualAcuityDataset(image_dir, k), batch_size=batch_size)):
                pass

            end_time = time.time()
            total_time += (end_time - start_time)

        avg_time = total_time / repetitions
        acuity_results.append(avg_time)
        print(f"Acuity | k={k}, Avg Time: {avg_time:.5f}s")

    for k in x_months_contrast:
        total_time = 0
        for _ in range(repetitions):
            start_time = time.time()

            for batch_id, (images, ages) in enumerate(DataLoader(VisualContrastDataset(image_dir, k), batch_size=batch_size)):
                pass

            end_time = time.time()
            total_time += (end_time - start_time)

        avg_time = total_time / repetitions
        contrast_results.append(avg_time)
        print(f"Contrast | k={k}, Avg Time: {avg_time:.5f}s")

    plt.figure(figsize=(12, 8))

    plt.plot(x_months_acuity, acuity_results, marker='o', color='blue', label='Acuity (Blur)')
    plt.plot(x_months_contrast, contrast_results, marker='o', color='red', label='Contrast')

    plt.xlabel('Age in months')
    plt.ylabel('Average Loading Time (s)')
    plt.title('Performance Evaluation: Acuity vs. Contrast')
    plt.grid(True)
    plt.legend()

    plt.show()

def show_images(dataset, dataloader):
    total_filter_time = 0

    for batch_id, (images, ages) in enumerate(dataloader):

        # Display images
        plt.figure(figsize=(math.ceil(math.sqrt(batch_size)), math.ceil(math.sqrt(batch_size))))

        for i in range(batch_size):
            plt.subplot(math.ceil(math.sqrt(batch_size)), math.ceil(math.sqrt(batch_size)), i + 1)
            plt.imshow(np.clip(images[i], 0, 1))
            plt.axis('off')

        margin = 0.05
        plt.subplots_adjust(wspace=margin, hspace=margin)
        plt.tight_layout(pad=1)

        plt.show()

        total_filter_time += dataset.get_total_filter_time()

    log_to_file(f"Total filter time (for all datasets): {total_filter_time:.5f} seconds")

# Main Script
image_directory = './sampleDataset2/'
age_months = 0
batch_size = 100

dataset = VisualContrastDataset(image_dir=image_directory, age_months=age_months)
dataset = VisualAcuityDataset(image_dir=image_directory, age_months=age_months)

results = []

show_images(VisualAcuityDataset(image_directory, age_months), DataLoader(VisualAcuityDataset(image_directory, age_months), batch_size=batch_size))

test_performance_comparison(image_directory, batch_size)