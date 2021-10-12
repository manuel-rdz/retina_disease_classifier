import numpy as np
import pandas as pd

import os
import cv2
import random

from sklearn.datasets import make_multilabel_classification

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
    return X, y

class MLSMOTE:
    def __init__(self,k):
        self.k=k
        self.full_label_set = []
        self.labels=[]
        self.features=[]

    def fit_resample(self,X,y,k):
        self.full_label_set = np.unique(np.array([a for x in y for a in (x if isinstance(x, list) else [x])]))
        self.labels=np.array([np.array(xi) for xi in y])
        self.features=X
        X_synth=[]
        y_synth=[]
        append_X_synth=X_synth.append
        append_y_synth=y_synth.append
        mean_ir=self.get_mean_imbalance_ratio()
        for label in self.full_label_set:
            irlbl=self.get_imbalance_ratio_per_label(label)
            if irlbl > mean_ir:
                min_bag=self.get_all_instances_of_label(label)
                for sample in min_bag:
                    distances=self.calc_distances(sample,min_bag)
                    distances=np.sort(distances,order='distance')
                    neighbours=distances[:k]
                    ref_neigh=np.random.choice(neighbours,1)[0]
                    X_new,y_new=self.create_new_sample(sample,ref_neigh[1],[x[1] for x in neighbours])
                    append_X_synth(X_new)
                    append_y_synth(y_new)
       
        return np.array(X_synth),np.array(y_synth)

    def create_new_sample(self,sample_id,ref_neigh_id,neighbour_ids):
        sample=self.features[sample_id]
        sample_labels=self.labels[sample_id]
        synth_sample=np.zeros(sample.shape[0])
        ref_neigh=self.features[ref_neigh_id]
        neighbours_labels=[]
        for ni in neighbour_ids:
            neighbours_labels.append(self.labels[ni].tolist())
        for i in range(synth_sample.shape[0]):
            #if f is numeric todo:implement nominal support
            diff=ref_neigh[i]-sample[i]
            offset=diff*random.uniform(0,1)
            synth_sample[i]=sample[i]+offset
        labels=sample_labels.tolist()
        labels+=[a for x in neighbours_labels for a in (x if isinstance(x, list) else [x])]
        labels=list(set(labels))
        head_index=int((self.k+ 1)/2)
        y=labels[:head_index]
        X=synth_sample
        return X,y


    def calc_distances(self,sample,min_bag):
        distances=[]
        append_distances=distances.append
        for bag_sample in min_bag:
            #if f is numeric todo:implement nominal support
            append_distances((np.linalg.norm(self.features[sample]-self.features[bag_sample]),bag_sample))
        dtype =  np.dtype([('distance', float), ('index', int)])
        return np.array(distances,dtype=dtype)

    def get_all_instances_of_label(self,label):
        instance_ids=[]
        append_instance_id=instance_ids.append
        for i,label_set in enumerate(self.labels):
            if label in label_set:
                append_instance_id(i)
        return np.array(instance_ids)

    def get_mean_imbalance_ratio(self):
        ratio_sum=np.sum(np.array(list(map(self.get_imbalance_ratio_per_label,self.full_label_set))))
        return ratio_sum/self.full_label_set.shape[0]

    def get_imbalance_ratio_per_label(self,l):
        sum_array=list(map(self.sum_h,self.full_label_set))
        sum_array=np.array(sum_array)
        return sum_array.max()/self.sum_h(l)

    def sum_h(self,l): 
        h_sum=0
        def h(l,Y):
            if l in Y:
                return 1
            else:
                return 0

        for label_set in self.labels:
            h_sum+=h(l,label_set)
        return h_sum

  
    def get_value_counts(self,labels):
        count_map=np.array(np.unique(labels, return_counts=True)).T
        counts=np.array([x[1] for x in count_map])
        return counts


if __name__=='__main__':

    x_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\all-images'
    y_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\STARE\\stare_dataset.csv'


    y = pd.read_csv(y_path)
    x = np.empty(0)
    dim = (384, 384)
    n_samples = 100

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
    mlsmote = MLSMOTE(k=5)
    x_res, y_res = mlsmote.fit_resample(x, y, 5)


    print(x.shape)
    print(y.shape)

    print(y.sum(axis=0))
    #print(y.sum(axis=1) > 1)

    #print(x_sub.shape)
    #print(y_sub.shape)

    print(y_res.sum(axis=0))
    print(y_res.shape)
    print(x_res.shape)

    #print(x_res.head())

"""
x, y = create_dataset()

print(x.shape)
print(y.shape)

print(x[:5, :])
print(y[:5, :])

print(y.sum(axis=0))

mlsmote = MLSMOTE(k=5)
x_res, y_res = mlsmote.fit_resample(x, y, 5)

print(x_res.shape)
print(y_res.shape)

print(y_res.sum(axis=0))"""

