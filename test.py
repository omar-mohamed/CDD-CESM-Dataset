from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from model_utils import set_gpu_usage, get_multilabel_evaluation_metrics, get_generator, get_evaluation_metrics
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import os

FLAGS = argHandler()
FLAGS.setDefaults()

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()


train_generator = get_generator(FLAGS.train_csv,FLAGS)
test_generator = get_generator(FLAGS.test_csv,FLAGS)

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)

def get_metrics_from_generator(generator,threshold=0.5, verbose=1):
    y_hat = visual_model.predict(generator, steps=generator.steps, workers=FLAGS.generator_workers,
                                           max_queue_size=FLAGS.generator_queue_length, verbose=verbose)
    y = generator.get_y_true()
    if FLAGS.multi_label_classification:
        get_multilabel_evaluation_metrics(y_hat, y, FLAGS.classes, threshold=threshold,image_names=generator.get_images_names(),save_path=os.path.join(FLAGS.save_model_path,'exact_match.csv'))
    else:
        y_hat = y_hat.argmax(axis=1)
        get_evaluation_metrics(y_hat, y, FLAGS.classes)

if FLAGS.multi_label_classification:
    visual_model.compile(loss='binary_crossentropy',
                         metrics=[metrics.BinaryAccuracy(threshold=FLAGS.multilabel_threshold)])
else:
    visual_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("***************Train Metrics*********************")
get_metrics_from_generator(train_generator, FLAGS.multilabel_threshold)
print("***************Test Metrics**********************")
get_metrics_from_generator(test_generator, FLAGS.multilabel_threshold)

