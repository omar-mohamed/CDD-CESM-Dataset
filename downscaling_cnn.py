from __future__ import absolute_import, division

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


# method to return the dense classifier
def get_downscaling_model(input_shape, downscaling_factor, num_filters=64):
    if downscaling_factor == 0:
        return None
    model = Sequential(name='Downscaling_Model')
    for i in range(downscaling_factor):
        if i==0:
            model.add(Conv2D(num_filters, (4, 4), strides=(2, 2), padding="same", activation="relu", input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, (4, 4), strides=(2, 2), activation="relu", padding="same"))
        model.add(Conv2D(num_filters, (3, 3), padding="same", activation="relu"))

    model.add(Conv2D(3, (3, 3), padding="same", activation="tanh"))

    return model
