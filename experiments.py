import pandas as pd
import numpy as np

def calculating_class_weights(y_true):
    y_pos = np.sum(y_true, axis=0)
    weights = y_true.shape[0] / y_pos

    return np.array(weights)


y_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\clean_merged_dataset_all.csv'

y = pd.read_csv(y_path)

y = y.iloc[:, 4:]

print(calculating_class_weights(y))