import pandas as pd
import numpy as np

def convert_to_labels(vec):
    labels = []
    for i, x in enumerate(vec):
        if(x == 1):
            labels.append(id_to_label[i])
    return labels

def multi_hot_embedding(labels):
    embedding = np.zeros(len(all_classes))
    for label in labels:
        id = label_to_id[label]
        embedding[id] = 1
    return embedding

def get_all_classes(dataset):
    # would be better if i had the classes saved somewhere
    # iterating through the dataset is suboptimal
    my_set = set()
    for idx, x in dataset.iterrows():
        labels = x.labels.split(',')
        my_set.update(labels)
    return my_set

train_curated = pd.read_csv("./input/train_curated.csv")
all_classes = get_all_classes(train_curated)
label_to_id = dict([(x,i) for i, x in enumerate(all_classes)])
id_to_label = dict([(i,x) for i, x in enumerate(all_classes)])