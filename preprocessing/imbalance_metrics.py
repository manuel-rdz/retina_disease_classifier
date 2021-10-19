import pandas as pd
import numpy as np

from resampling import metrics, lp_rus, lp_ros, ml_ros, ml_rus, mlsmote

import os
import cv2

def get_imbalance_metrics(y):
    ir_per_label = []
    for label in range(y.shape[1]):
        ir_per_label.append(metrics.ir_per_label(label, y))

    mean_ir = metrics.mean_ir(y)
    coeff_ir = metrics.cvir(y)

    return np.concatenate((ir_per_label, [mean_ir], [coeff_ir]))

def resample(x, y, resample_func, oversample, percentage=10, k=5, mlsmote=False):
    if mlsmote:
        x_new, y_new = resample_func(x, y, k)
        return np.append(x, x_new, axis=0), np.append(y, y_new, axis=0)
    else:
        idxs = resample_func(y, percentage)

        if oversample:
            return np.append(y, y[idxs], axis=0)
        else:
            mask = np.ones(y.shape[0], dtype=bool)
            mask[idxs]=False
            return y[mask]


x_path = ['C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop',
'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\ARIA\\all_images_crop',
'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training']

y_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\clean_merged_dataset_all.csv'

output_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets'


y = pd.read_csv(y_path)
x = np.empty(0)
dim = (384, 384)

start_col = 4

# create the x dataset flattening the images
for path in x_path:
    for idx, f in enumerate(os.listdir(path)):
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path)
        img_rsz = cv2.resize(img, dim)
        flatten_img = img_rsz.flatten()
        #show_img(pd.DataFrame(flatten_img))
        if len(x) == 0:
            x = flatten_img
            x = x[np.newaxis, ...]
        else:
            x = np.append(x, [flatten_img], axis=0)

# create the output file
output = y.iloc[:, start_col:].columns.values
output = np.append(output, ['Mean IR', 'Coeff IR'])

header = ['Labels']

y = y.iloc[:x.shape[0], start_col:].to_numpy(dtype=np.int32)


header.append('Baseline')
output = np.column_stack((output, get_imbalance_metrics(y)))

header.append('LP_ROS')
y_new = resample(x, y, lp_ros.LP_ROS, True)
output = np.column_stack((output, get_imbalance_metrics(y_new)))

header.append('LP_RUS')
y_new = resample(x, y, lp_rus.LP_RUS, False)
output = np.column_stack((output, get_imbalance_metrics(y_new)))

header.append('LP_RUS + LP_ROS')
y_new = resample(x, y, lp_rus.LP_RUS, False)
y_new = resample(x, y_new, lp_ros.LP_ROS, True)
output = np.column_stack((output, get_imbalance_metrics(y_new)))

header.append('ML_ROS')
y_new = resample(x, y_new, ml_ros.ML_ROS, True)
output = np.column_stack((output, get_imbalance_metrics(y_new)))

header.append('ML_RUS')
y_new = resample(x, y, ml_rus.ML_RUS, False)
output = np.column_stack((output, get_imbalance_metrics(y_new)))

header.append('ML_RUS + ML_ROS')
y_new = resample(x, y, ml_rus.ML_RUS, False)
y_new = resample(x, y_new, ml_ros.ML_ROS, True)
output = np.column_stack((output, get_imbalance_metrics(y_new)))

header.append('MLSMOTE')
x_new, y_new = resample(x, y, mlsmote.MLSMOTE, True, mlsmote=True)
output = np.column_stack((output, get_imbalance_metrics(y_new)))


np.savetxt(os.path.join(output_path, 'imb_algo_comparison.csv'),
    output,
    header=",".join(map(str, header)),
    delimiter=', ',
    fmt='% s',
    comments='')
