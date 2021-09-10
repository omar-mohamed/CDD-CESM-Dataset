import pandas as pd
import os
import numpy as np

dataset_df = pd.read_csv('./data/all_data.csv')

mapping = [0,1,2,2,3,3]
new_labels = []
for _, row in dataset_df.iterrows():
    labels = row['BIRADS']
    print(labels)
    labels = labels.split('$')
    for i in range(len(labels)):
        labels[i] = str(mapping[int(labels[i])])
    labels = list(set(labels))
    labels = "$".join(labels)
    print(labels)
    print("-----------------")
    new_labels.append(labels)

dataset_df['BIRADS'] = new_labels

dataset_df.to_csv('./data/all_data_3c.csv',index=False)