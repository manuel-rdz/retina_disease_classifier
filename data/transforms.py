import albumentations as a
import numpy as np

from albumentations.pytorch import ToTensorV2


def get_resnet_train_transforms(image_size):
    return a.Compose([
        a.Resize(image_size, image_size),
        a.RandomCrop(448, 448, always_apply=True),
        a.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_resnet_valid_transforms(image_size):
    return a.Compose([
        a.Resize(image_size, image_size),
        a.CenterCrop(448, 448, always_apply=True),
        a.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_vgg16_train_transforms(image_size):
    transforms_train = a.Compose([
        a.Resize(image_size, image_size),
        a.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    return transforms_train


def get_riadd_train_transforms(image_size):
    transforms_train = a.Compose([
        #a.RandomResizedCrop(image_size, image_size, scale=(0.85, 1), p=1),
        a.Resize(image_size, image_size),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=30),
        a.MedianBlur(blur_limit = 7, p=0.3),
        a.GaussNoise(var_limit=(0,0.15*255), p = 0.5),
        a.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        a.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.3),
        a.Cutout(max_h_size=20, max_w_size=20, num_holes=5, p=0.5),
        a.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return transforms_train


def get_riadd_valid_transforms(image_size):
    valid_transforms = a.Compose([
        a.Resize(image_size, image_size),
        a.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    return valid_transforms


def get_riadd_test_transforms(image_size, use_tta):
    if use_tta: 
        return a.Compose([
            a.Resize(image_size, image_size),
            a.HorizontalFlip(p=0.5),
            a.Rotate(limit=30),
            a.HueSaturationValue(hue_shift_limit=10,   sat_shift_limit=10, val_shift_limit=10, p=0.5),
            a.RandomBrightnessContrast(brightness_limit=(-0.2,0.2),contrast_limit=(-0.2, 0.2), p=0.5),
            a.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    return get_riadd_valid_transforms(image_size)