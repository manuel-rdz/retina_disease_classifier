import timm
import torch.nn as nn

print(timm.list_models('*beit*', pretrained=True))

#model = timm.create_model('efficientnet_b3', pretrained=False)
#print(model)

#model.fc = nn.Linear(model.fc.in_features, 20)
#print(model)

'''
Calculation based on 
https://medium.com/analytics-vidhya/understanding-the-vision-transformer-and-counting-its-parameters-988a4ea2b8f3

p = 16
c = 3
d = 2048
n = 576
n_classes = 20

l = 1
k = 4
d_mlp = 2048
d_h = 512

#backbone
ResNet101 = 44500000
ResNet50 = 25600000

backbone = ResNet50

#patch_embedding
patch_embedding = p**2*c*d + (n+1+n_classes)*d
print(patch_embedding)

#transformer_encoder
transformer_encoder = l*(k*d*3*d_h + d*d + d*d_mlp + d_mlp*d)
print(transformer_encoder)

#fine_tuning
fine_tuning = d*n_classes
print(fine_tuning)

print('model_size: ', patch_embedding + transformer_encoder + fine_tuning + backbone)
'''

'''import pandas as pd
import numpy as np
from resampling import ml_ros

def get_class_weights(y_true):
    y_pos = np.sum(y_true, axis=0)
    max_samples = y_pos.max()
    weights = max_samples / y_pos

    return np.array(weights)


data = pd.read_csv('C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\drop_all\\20_labels\\merged_20_labels_drop_10.0_perc.csv')

y = data.iloc[:, 4:].to_numpy()

copy_idxs = ml_ros.ML_ROS(y, 10)

y_new = np.append(y, y[copy_idxs, :], axis=0)

print(get_class_weights(y))
print(get_class_weights(y_new))'''