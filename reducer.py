import os
import shutil
import random
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def reduce_tiny_imagenet(root_dir, num_classes=20, seed=42):
    random.seed(seed)

    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')  # Path to validation images
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    all_classes = os.listdir(train_dir)
    if '.DS_Store' in all_classes:
        all_classes.remove('.DS_Store')

    random.shuffle(all_classes)
    selected_classes = all_classes[:num_classes]

    reduced_train_dir = os.path.join(root_dir, 'train_reduced')
    reduced_val_dir = os.path.join(root_dir, 'val_reduced') # Reduced validation folder
    reduced_val_images_dir = os.path.join(reduced_val_dir, 'images') # Images are in a subfolder
    os.makedirs(reduced_train_dir, exist_ok=True)
    os.makedirs(reduced_val_dir, exist_ok=True)
    os.makedirs(reduced_val_images_dir, exist_ok=True)

    # Copy training data (correctly using copytree)
    for class_name in selected_classes:
        src_train_class_dir = os.path.join(train_dir, class_name)
        dst_train_class_dir = os.path.join(reduced_train_dir, class_name)
        if os.path.exists(src_train_class_dir):
            try:
                shutil.copytree(src_train_class_dir, dst_train_class_dir)
            except FileExistsError:
                pass
        else:
            print(f"Warning: Training class directory {src_train_class_dir} not found.")

    # Copy validation data (Corrected: Preserve original structure)
    reduced_val_annotations_file = os.path.join(reduced_val_dir, 'val_annotations.txt')
    with open(val_annotations_file, 'r') as in_f, open(reduced_val_annotations_file, 'w') as out_f:
        for line in in_f:
            parts = line.split('\t')
            image_name = parts[0]
            image_class = parts[1]
            if image_class in selected_classes:
                src_image_path = os.path.join(val_images_dir, image_name)
                dst_image_path = os.path.join(reduced_val_images_dir, image_name)
                shutil.copy2(src_image_path, dst_image_path)
                out_f.write(line) # Write line to the new annotation file
    print(f"Reduced dataset created in {root_dir}")

root_dir = './tiny-imagenet-200'  # Replace with the actual path
reduce_tiny_imagenet(root_dir, num_classes=20)
