import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import Sequence
from PIL import Image
from skimage.transform import resize


class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, label_columns, multi_label_classification, class_names, source_image_dir,
                 batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=1):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.label_columns = label_columns
        self.multi_label_classification = multi_label_classification
        self.class_counts = [0] * len(class_names)

        self.current_step = -1
        self.prepare_dataset()
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sides = self.side[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = zip(batch_x_path, batch_sides)
        batch_x = np.asarray([self.load_image(x_path, side) for x_path, side in batch])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_file, side):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        if self.multi_label_classification:
            return self.y[:self.steps * self.batch_size, :]
        else:
            return self.y[:self.steps * self.batch_size]

    def get_class_counts(self):
        return self.class_counts

    def get_sparse_labels(self, y):
        labels = np.zeros(y.shape[0], dtype=int)
        index = 0

        for label in y:
            labels[index] = self.class_names.index(label)
            self.class_counts[labels[index]] += 1
            index += 1

        return labels

    def get_onehot_labels(self, y):
        onehot = np.zeros((y.shape[0], len(self.class_names)))
        index = 0
        for label in y:
            labels = str(label[0]).split("$")
            for l in labels:
                ind = self.class_names.index(l)
                onehot[index, ind] = 1
            index += 1
        return onehot

    def convert_labels_to_numbers(self, y):
        if self.multi_label_classification:
            return self.get_onehot_labels(y)
        else:
            return self.get_sparse_labels(y)

    def get_images_names(self):
        return self.image_names

    def get_images_path(self, image_names):
        for i in range(image_names.shape[0]):
            image_names[i] = image_names[i].strip() + '.jpg'
        return image_names

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y, self.side, self.image_names = self.get_images_path(df["Image_name"].values), \
                                                           self.convert_labels_to_numbers(
                                                               df[self.label_columns].values), df['Side'].values, df[
                                                               'Image_name'].values

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
