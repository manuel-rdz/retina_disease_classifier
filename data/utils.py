from data.datasets import MergedDataset, RIADDDataset
import data.transforms as t


def get_dataset(df_data, img_path, transforms, start_col):
    if len(img_path) == 1:
        return RIADDDataset(image_ids=df_data, img_path=img_path[0], transform=transforms)
    else:
        return MergedDataset(image_ids=df_data, img_path=img_path, transform=transforms, start_col_labels=start_col)


def get_transformations(transforms, img_size):
    if transforms == 'vgg16':
        train_transforms = t.get_vgg16_train_transforms(img_size)
        val_transforms = t.get_riadd_valid_transforms(img_size)
    elif transforms == 'resnet':
        train_transforms = t.get_resnet_train_transforms(img_size)
        val_transforms = t.get_resnet_valid_transforms(img_size)
    elif transforms == 'efficientnet':
        train_transforms = t.get_efficientnet_gray_train_transforms(img_size)
        val_transforms = t.get_efficientnet_gray_valid_transforms(img_size)
    else:
        train_transforms = t.get_riadd_train_transforms(img_size)
        val_transforms = t.get_riadd_valid_transforms(img_size)

    return train_transforms, val_transforms
