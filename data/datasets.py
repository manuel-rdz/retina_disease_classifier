import os
import cv2
import numpy as np
import torch
import torch.utils.data as data


# Training / Testing
class RIADDDataset(data.Dataset):
    def __init__(self, image_ids, img_path='', transform=None, only_disease=False):

        self.image_ids = image_ids
        self.img_path = img_path
        self.transform = transform
        self.only_disease = only_disease
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        imgId = str(imgId) + '.png'
        if self.only_disease == True:
            label_2 = self.image_ids.iloc[index, 2:].values.astype(np.int64)
            label_2 = sum(label_2)
            label = self.image_ids.iloc[index, 1:2].values.astype(np.int64)
            if label_2 > 0:
                label = np.append(label, 1)
            else:
                label = np.append(label,0)
        else:
            label = self.image_ids.iloc[index, 1:].values.astype(np.int64)

        imgpath = os.path.join(self.img_path, imgId)
        img = cv2.imread(imgpath)
        #img = crop_maskImg(img)
        img = img[:, :, ::-1]
        img = self.transform(image = img)['image']
        return img, label


class MergedDataset(data.Dataset):  # for training/testing
    def __init__(self, image_ids, img_path=None, transform=None, only_disease=False, start_col_labels=3):

        self.image_ids = image_ids
        self.img_path = img_path
        self.transform = transform
        self.only_disease = only_disease
        self.start_col_labels = start_col_labels
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        if self.image_ids.iloc[index, 1] == 1: # ARIA
            imgId = str(imgId) + '.tif'
            dataset_idx = 0
        elif self.image_ids.iloc[index, 2] == 1: # STARE
            imgId = str(imgId) + '.png'
            dataset_idx = 1
        elif self.image_ids.iloc[index, 3] == 1: # RFMiD
            imgId = str(imgId) + '.png'
            dataset_idx = 2
        else: # SYTHETIC
            imgId = str(imgId) + '.png'
            dataset_idx = 3

        label = self.image_ids.iloc[index, self.start_col_labels:].values.astype(np.int64)
        imgpath = os.path.join(self.img_path[dataset_idx], imgId)
        img = cv2.imread(imgpath)
        try:
            img = img[:, :, ::-1]
        except:
            print(imgpath)
        img = self.transform(image = img)['image']
        return img, label
