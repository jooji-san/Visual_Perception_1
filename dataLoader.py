import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from enum import Flag, auto, Enum

class TransformationType(Flag):
    CONTRAST = auto()
    ACUITY = auto()

class SplitType(Enum):
    TRAIN = auto()
    VAL = auto()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def age_to_acuity_scale(months):
    age_data_acuity = np.array([0, 1, 3, 6, 12, 24, 36, 48])
    # 0 month should not be 687 as shown in the data, but 600 as shown in the exercise description
    acuity_data = np.array([600, 581, 231, 109, 66, 36, 25, 20])
    return np.interp(months, age_data_acuity, acuity_data) / 20

def apply_acuity_transform(image, age_months):
    max_blur_radius = 4
    acuity_scale = age_to_acuity_scale(age_months)
    blur_radius = max_blur_radius * (acuity_scale - (20 / 20)) / ((687 / 20) - (20 / 20))

    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    return blurred_image

def age_to_contrast_scale(months):
    age_data_contrast = np.array([1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 216])
    contrast_data = np.array(
        [0.054092562, 0.079797917, 0.087848712, 0.197312866, 0.301349368, 0.548930981, 0.64538004, 0.635222933,
         0.675745399, 0.772787744, 0.824422479, 0.938288057, 1], dtype=float)

    return np.interp(months, age_data_contrast, contrast_data)


def apply_contrast_transform(image, contrast_scale):
    contrast_enhancer = ImageEnhance.Contrast(image)
    contrasted_image = contrast_enhancer.enhance(contrast_scale)

    return contrasted_image

class DatasetWithTransformationByAge(Dataset):
    def __init__(self, root_dir, split_type, age_months, transform_type):
        self.root_dir = root_dir

        # Create a shared class_to_idx dictionary
        self.class_to_idx = {}

        if split_type == SplitType.TRAIN:
            self.dir = os.path.join(root_dir, "train_reduced")
            self.classes = [d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))]

            # Populate the class_to_idx dictionary
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            self.samples = []
            for class_name in self.classes:
                class_dir = os.path.join(self.dir, class_name, "images")
                class_idx = self.class_to_idx[class_name]

                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
        elif split_type == SplitType.VAL:
            self.dir = os.path.join(root_dir, "val_reduced")
            self.classes = [d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))]
            self.samples = []
            with open(os.path.join(self.dir, "val_annotations.txt"), "r") as f:
                for line in f:
                    img_name, class_name = line.split("\t")[:2]
                    img_path = os.path.join(self.dir, "images", img_name)
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.class_to_idx)
                    self.samples.append((img_path, self.class_to_idx[class_name]))


        self.transformType = transform_type
        self.age_months = age_months


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transformType is None:
            pass
        else:
            if TransformationType.ACUITY in self.transformType:
                image = apply_acuity_transform(image, self.age_months)
            if TransformationType.CONTRAST in self.transformType:
                image = apply_contrast_transform(image, age_to_contrast_scale(self.age_months))

        return transform(image), label

