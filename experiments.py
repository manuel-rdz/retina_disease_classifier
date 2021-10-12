import numpy as np
import pandas as pd

import random
import os
import cv2
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale


def create_dataset(n_sample=1000):
    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function
    
    args
    nsample: int, Number of sample to be created
    
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_multilabel_classification(n_classes=5, n_features=10, n_labels=2, n_samples=1000, random_state=10)

    #y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), pd.DataFrame(y)

def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.0]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label

def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target

def show_img(img):
    img = img.to_numpy(dtype=np.uint8)
    print(img.dtype)

    img = minmax_scale(img, feature_range=(0,255))
    img = img.reshape((384, 384),)
    img = img.astype('uint8')

    print(img.dtype)

    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__=='__main__':

    x_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all-images'
    y_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\stare_dataset.csv'


    y = pd.read_csv(y_path)
    x = np.empty(0)
    dim = (384, 384)

    for idx, f in enumerate(os.listdir(x_path)):
        img_path = os.path.join(x_path, f)
        img = cv2.imread(img_path)[:,:,1]
        img_rsz = cv2.resize(img, dim)

        flatten_img = img_rsz.flatten()
        #show_img(pd.DataFrame(flatten_img))

        if len(x) == 0:
            x = flatten_img
            x = x[np.newaxis, ...]
        else:
            x = np.append(x, [flatten_img], axis=0)
        #break

    x = pd.DataFrame(x)
    y = y.iloc[:, 2:]

    #print(x.shape)
    #print(x.head())
    #print(y.head())

    x_sub, y_sub = get_minority_samples(x, y)   #Getting minority instance of that datframe
    x_res, y_res = MLSMOTE(x_sub, y_sub, 5)     #Applying MLSMOTE to augment the dataframe

    for i in range(x_res.shape[0]):
        show_img(x_res.iloc[i])

    #print(x.iloc[0])
    show_img(x.iloc[0])

    #print(x.shape)
    #print(y.shape)

    #print(y.sum(axis=0))
    #print(y.sum(axis=1) > 1)

    #print(x_sub.shape)
    #print(y_sub.shape)

    #print(y_res.sum(axis=0))
    #print(y_res.shape)
    #print(x_res.shape)

    #print(y_res)
    #print(x_res.head())

    #print(x_res.head())
"""
X, y = create_dataset()  # Creating a Dataframe
X_sub, y_sub = get_minority_samples(X, y)  # Getting minority samples of that datframe
X_res, y_res = MLSMOTE(X_sub, y_sub, 50, 5)  # Applying MLSMOTE to augment the dataframe

print(X.head())
print(y.head())

print(X.shape)
print(y.shape)

print(y.sum(axis=0))

#print(X_sub.shape)
#print(y_sub.shape)

print(y_res.sum(axis=0))
#print(X_res.shape)
"""