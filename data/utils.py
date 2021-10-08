from data.datasets import MergedDataset, RIADDDataset


def get_dataset(df_data, img_path, transforms, start_col):
    if len(img_path) == 1:
        return RIADDDataset(image_ids=df_data, img_path=img_path[0], transform=transforms)
    else:
        return MergedDataset(image_ids=df_data, img_path=img_path, transform=transforms, start_col_labels=start_col)
