import torch
from torchvision.io import read_image, write_png
import matplotlib.pyplot as plt
import traceback


def create_frequency_masks(size):
    """Create frequency band masks using PyTorch"""
    rows, cols = size
    center_row, center_col = rows // 2, cols // 2

    y_grid, x_grid = torch.meshgrid(
        torch.arange(rows, dtype=torch.float32),
        torch.arange(cols, dtype=torch.float32),
        indexing='ij'
    )

    # Calculate distance from center
    distance = torch.sqrt(
        (y_grid - center_row) ** 2 + (x_grid - center_col) ** 2
    ) / (min(rows, cols) / 2)

    # Define frequency bands
    low_mask = (distance <= 0.2).float()
    mid_mask = ((distance > 0.2) & (distance <= 0.5)).float()
    high_mask = (distance > 0.5).float()

    return [low_mask, mid_mask, high_mask]


def separate_frequencies(image):
    """Separate image into different frequency bands using FFT"""
    # Ensure image is on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)

    # Apply FFT
    fft = torch.fft.fft2(image.float())
    fft_shift = torch.fft.fftshift(fft)

    # Create frequency masks
    masks = [mask.to(device) for mask in create_frequency_masks(image.shape[-2:])]

    # Apply masks and inverse FFT
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

def process_image(image_path):
    # Read image
    image = read_image(image_path).float() / 255.0

    # Separate frequencies for each channel
    frequency_bands = []
    for c in range(image.shape[0]):
        channel_bands = separate_frequencies(image[c])
        frequency_bands.append(channel_bands)

    # Apply contrast sensitivity
    sensitivity_factors = [1.5, 0.8, 2]
    transformed_bands = []

    for channel_idx, channel_bands in enumerate(frequency_bands):
        transformed_channel = []
        for band_idx, (band, factor) in enumerate(zip(channel_bands, sensitivity_factors)):
            band = band.unsqueeze(0)
            print(f"Channel {channel_idx}, Band {band_idx} BEFORE adjust_contrast:",
                  "min:", torch.min(band).item(),
                  "max:", torch.max(band).item(),
                  "mean:", torch.mean(band).item(),
                  "factor:", factor)

            transformed = adjust_contrast(band, factor)

            print(f"Channel {channel_idx}, Band {band_idx} AFTER adjust_contrast:",
                  "min:", torch.min(transformed).item(),
                  "max:", torch.max(transformed).item(),
                  "mean:", torch.mean(transformed).item())

            print(f"Channel {channel_idx}, Band {band_idx} are exactly equal:",
                  torch.all(band == transformed).item())
            print("---")

            transformed = transformed.squeeze(0)
            transformed_channel.append(transformed)
        transformed_bands.append(transformed_channel)

    # Combine bands for each channel
    result_channels = []
    for transformed_channel in transformed_bands:
        result_channel = torch.sum(torch.stack(transformed_channel), dim=0)
        result_channels.append(result_channel)

    # Stack channels back together
    result = torch.stack(result_channels)

    # Normalize and convert back to uint8
    result = torch.clamp(result, 0, 1) * 255
    result = result.to(torch.uint8)

    return result


def save_image(image, output_path):
    """Save the processed image"""
    result = torch.clamp(image, 0, 1) * 255
    result = result.to(torch.uint8)
    write_png(result, output_path)


def visualize_images(original, processed):
    """Visualize the original and processed images"""
    plt.figure(figsize=(10, 5))

    # Convert tensors to numpy arrays and permute dimensions for color images
    original = original.permute(1, 2, 0).numpy()
    processed = processed.permute(1, 2, 0).numpy()

    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(processed)
    plt.title('Processed')
    plt.axis('off')

    plt.show()


# Example usage
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Process image
    image_path = 'my_image.jpg'
    original = read_image(image_path).float() / 255.0
    try:
        # Separate frequencies and apply contrast sensitivity
        result = process_image(image_path)

        # Save result
        save_image(result, 'processed_image.png')

        # Visualize original and result
        visualize_images(original, result)

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()