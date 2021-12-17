from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from utils import set_gpu_usage, get_multilabel_evaluation_metrics, get_generator, get_evaluation_metrics
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import os
import numpy as np
from gradcam import GradCAM
import cv2
from tqdm import tqdm


FLAGS = argHandler()
FLAGS.setDefaults()

write_path = os.path.join(FLAGS.save_model_path,'cam_output')

try:
    os.makedirs(write_path)
except:
    print("path already exists")

set_gpu_usage(FLAGS.gpu_percentage)

model_factory = ModelFactory()

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)
FLAGS.batch_size = 1
test_generator = get_generator(FLAGS.train_csv,FLAGS)

images_names = test_generator.get_images_names()



for batch_i in tqdm(range(test_generator.steps)):
    batch, _ = test_generator.__getitem__(batch_i)
    image_path = os.path.join(FLAGS.image_directory, images_names[batch_i])
    original = cv2.imread(image_path)
    preds = visual_model.predict(batch)
    predicted_class = np.argmax(preds[0])
    label = FLAGS.classes[predicted_class]
    cam = GradCAM(visual_model, predicted_class)
    heatmap = cam.compute_heatmap(batch)

    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, original, alpha=0.5)

    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(write_path,images_names[batch_i]),output)

