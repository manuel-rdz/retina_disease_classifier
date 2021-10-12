# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import random
import cv2
import os
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import NearestNeighbors


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

def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe
    
    args
    df: pandas.DataFrame, target label df whose tail label has to identified
    
    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    print(columns)
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df.loc[:, columns[column]].value_counts()[0]
    irpl = max(irpl)/irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label

def get_index(df):
  """
  give the index of all tail_label rows
  args
  df: pandas.DataFrame, target label df from which index for tail label has to identified
    
  return
  index: list, a list containing index number of all the tail label
  """
  tail_labels = get_tail_label(df)
  index = set()
  for tail_label in tail_labels:
    sub_index = set(df[df[tail_label]==1].index)
    index = index.union(sub_index)
  return list(index)

def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels
    
    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    print(index)
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(X)
    euclidean,indices= nbs.kneighbors(X)
    return indices

def MLSMOTE(X,y, n_sample):
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
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0,n-1)
        neighbour = random.choice(indices2[reference,1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val>2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbour,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    #new_X = pd.concat([X, new_X], axis=0)
    #target = pd.concat([y, target], axis=0)
    return new_X, target

if __name__=='__main__':

    x_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all-images'
    y_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\stare_dataset.csv'


    y = pd.read_csv(y_path)
    x = np.empty(0)
    dim = (384, 384)
    n_samples = 200

    for idx, f in enumerate(os.listdir(x_path)):
        if idx == n_samples:
            break

        img_path = os.path.join(x_path, f)
        img = cv2.imread(img_path)
        img_rsz = cv2.resize(img, dim)

        flatten_img = img_rsz.flatten()

        if len(x) == 0:
            x = flatten_img
            x = x[np.newaxis, ...]
        else:
            x = np.append(x, [flatten_img], axis=0)

    x = pd.DataFrame(x)
    y = y.iloc[:n_samples, 1:]

    #print(x.shape)
    #print(x.head())
    #print(y.head())

    x_sub, y_sub = get_minority_instace(x, y)   #Getting minority instance of that datframe
    x_res, y_res = MLSMOTE(x_sub, y_sub, 15)     #Applying MLSMOTE to augment the dataframe




    print(x.shape)
    print(y.shape)

    print(y.sum(axis=0))
    #print(y.sum(axis=1) > 1)

    #print(x_sub.shape)
    #print(y_sub.shape)

    print(y_res.sum(axis=0))
    print(y_res.shape)
    print(x_res.shape)

    print(y_res)

    #print(x_res.head())

    """
    main function to use the MLSMOTE
    
    X, y = create_dataset()                     #Creating a Dataframe
    X_sub, y_sub = get_minority_instace(X, y)   #Getting minority instance of that datframe
    X_res,y_res =MLSMOTE(X_sub, y_sub, 0)     #Applying MLSMOTE to augment the dataframe




print(X.shape)
print(y.shape)

print(y.sum(axis=0))
#print(y.sum(axis=1) > 1)

#print(X_sub.shape)
#print(y_sub.shape)

print(y_res.sum(axis=0))
print(y_res.shape)"""