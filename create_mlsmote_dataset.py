import numpy as np
import pandas as pd
import os
import cv2

from sklearn.preprocessing import minmax_scale
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from resampling.mlsmote import MLSMOTE



if __name__=='__main__':

    x_paths = ['C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\ARIA\\all_images_crop',
     'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training',
     'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop']

    y_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\clean_merged_dataset_all.csv'

    output_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\synth_data'

    y = pd.read_csv(y_path)
    x = np.empty(0)
    
    dim = (384, 384)
    start_col = 4

    labels = y.columns.values
    #y = y.iloc[:, start_col:].to_numpy(dtype=np.int32)

    folds = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_i, (train_idx, val_idx) in enumerate(folds.split(y, y.iloc[:, start_col:])):

        print(len(train_idx))
        print(len(val_idx))
        print(val_idx.shape)

        train_x = y.iloc[train_idx, 0].to_numpy()
        train_y = y.iloc[train_idx, 1:].to_numpy(dtype=np.int32)

        mask = np.ones(y.shape[0], dtype=bool)
        mask[train_idx]=False

        val_x = y.iloc[mask, 0].to_numpy()
        val_y = y.iloc[mask, 1:].to_numpy(dtype=np.int32)

        print(labels)
        print(train_y.shape)
        print(val_y.shape)

        merged_idx = 0
        for x_path in x_paths:
            for f in os.listdir(x_path):
                if not merged_idx in train_idx:
                    merged_idx += 1
                    continue

                img_path = os.path.join(x_path, f)
                img = cv2.imread(img_path)
                img_rsz = cv2.resize(img, dim)

                flatten_img = img_rsz.flatten()

                if len(x) == 0:
                    x = flatten_img
                    x = x[np.newaxis, ...]
                else:
                    x = np.append(x, [flatten_img], axis=0)

                merged_idx += 1


        images, new_y = MLSMOTE(x, train_y[:, start_col-1:], 5)

        new_x = np.arange(new_y.shape[0])

        np.savetxt(os.path.join(output_path, 'mlsmote_labels.csv'),
        np.column_stack((new_x, new_y)),
        header=",".join(map(str, np.concatenate((['ID'], labels[start_col:])))),
        delimiter=', ',
        fmt='% s',
        comments='')

        np.savetxt(os.path.join(output_path, 'merged_train_labels.csv'),
        np.column_stack((train_x, train_y)),
        header=",".join(map(str, labels)),
        delimiter=', ',
        fmt='% s',
        comments='')

        np.savetxt(os.path.join(output_path, 'merged_val_labels.csv'),
        np.column_stack((val_x, val_y)),
        header=",".join(map(str, labels)),
        delimiter=', ',
        fmt='% s',
        comments='')

        os.makedirs(os.path.join(output_path, 'all_images'), exist_ok=True)

        for idx, img in enumerate(images):
            img = img.astype(np.uint8)

            img = minmax_scale(img, feature_range=(0,255))
            img = img.reshape((384, 384, 3),)
            img = img.astype('uint8')

            cv2.imwrite(os.path.join(output_path, 'all_images', str(idx)+'.png'),img)

        break


    #print(x.shape)
    #print(x.head())
    #print(y.head())

    #x_sub, y_sub = get_minority_samples(x, y)   #Getting minority instance of that datframe
    #x_res, y_res = MLSMOTE(x_sub, y_sub, 5)     #Applying MLSMOTE to augment the dataframe



    #for i in range(x_res.shape[0]):
    #    show_img(x_res.iloc[i])

    #print(x.iloc[0])
    #show_img(x.iloc[0])