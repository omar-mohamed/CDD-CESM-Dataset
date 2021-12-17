import pandas as pd
import os
import numpy as np
from configs import argHandler

FLAGS = argHandler()
FLAGS.setDefaults()

dataset_df = pd.read_excel('./data/annotations.xlsx')

test_set_fraction = 0.2

shuffle = True

if shuffle:
    dataset_df = dataset_df.sample(frac=1., random_state=np.random.randint(1, 100))


def get_sparse_labels(y):
    labels = np.zeros(y.shape[0], dtype=int)
    class_counts = np.zeros(3, dtype=int)
    mapping = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
    index = 0
    for label in y:
        try:
            labels[index] = mapping[label]
            class_counts[mapping[label]] += 1
        except:
            pass
        index += 1
    return labels, class_counts


def make_dict(dataset_df):
    dict = {}
    for column in dataset_df:
        dict[column] = []
    return dict


def add_row(dict, df_row):
    for key in dict.keys():
        dict[key].append(df_row[key])


def split_train_test(dataset_df):
    labels = dataset_df[FLAGS.csv_label_columns[0]]
    sparse_labels, class_counts = get_sparse_labels(labels)

    test_fraction_count = (class_counts * test_set_fraction).astype(int)
    print("Number of records for each class: {}".format(class_counts))
    print("Number of records for each class in test set: {}".format(test_fraction_count))

    train_dict = make_dict(dataset_df)
    test_dict = make_dict(dataset_df)
    test_count_so_far = np.zeros(3)
    index = 0
    for label in sparse_labels:
        if test_count_so_far[label] < test_fraction_count[label]:
            test_count_so_far[label] += 1
            add_row(test_dict, dataset_df.iloc[index])
        else:
            add_row(train_dict, dataset_df.iloc[index])
        index += 1

    return train_dict, test_dict


train_dict, test_dict = split_train_test(dataset_df)

training_df = pd.DataFrame(train_dict)
testing_df = pd.DataFrame(test_dict)

training_df.to_csv(os.path.join("./data", "train_set.csv"), index=False)

testing_df.to_csv(os.path.join("./data", "test_set.csv"), index=False)
