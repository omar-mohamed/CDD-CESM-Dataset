from __future__ import absolute_import, division

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianDropout, Flatten
from tensorflow.keras import regularizers


# method to return the dense classifier
def get_classifier(input_length, multi_label_classification, layer_sizes=[100], output_size=2):
    model = Sequential()
    model.add(Flatten())
    for layer_size in layer_sizes:
        if layer_size < 1:
            model.add(GaussianDropout(layer_size))
        else:
            model.add(Dense(layer_size, activation='relu'))

    if multi_label_classification:
        model.add(Dense(output_size, activation='sigmoid', name="predictions"))
    else:
        model.add(Dense(output_size, activation='softmax', name="predictions"))

    return model
