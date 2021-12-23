import albumentations
import numpy as np

from albumentations.pytorch import ToTensorV2

def get_riadd_train_transforms(image_size):
    transforms_train = albumentations.Compose([
        #albumentations.RandomResizedCrop(image_size, image_size, scale=(0.85, 1), p=1), 
        albumentations.Resize(image_size, image_size), 
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(limit=30),
        albumentations.MedianBlur(blur_limit = 7, p=0.3),
        albumentations.GaussNoise(var_limit=(0,0.15*255), p = 0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.3),
        albumentations.Cutout(max_h_size=20, max_w_size=20, num_holes=5, p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return transforms_train


def get_riadd_valid_transforms(image_size):
    valid_transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return valid_transforms


def get_riadd_test_transforms(image_size, use_tta):
    if use_tta: 
        return albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=30),
            albumentations.HueSaturationValue(hue_shift_limit=10,   sat_shift_limit=10, val_shift_limit=10, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2),contrast_limit=(-0.2, 0.2), p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    return get_riadd_valid_transforms(image_size)