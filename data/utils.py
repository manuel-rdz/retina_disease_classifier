from data.datasets import MergedDataset, RIADDDataset


def get_dataset(df_data, img_path, transforms):
    if len(img_path) == 1:
        return RIADDDataset(image_ids=df_data, img_path=img_path[0], transform=transforms)
    elif len(img_path) == 2:
        return MergedDataset(image_ids=df_data, img_path=img_path, transform=transforms)
