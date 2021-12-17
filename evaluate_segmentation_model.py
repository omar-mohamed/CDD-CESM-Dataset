from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from utils import get_generator, alpha_blend, get_segmented_image, is_covering_segmentation, apply_white_threshold, \
    get_IOU, get_f1, remove_corner_highlights
from tensorflow.keras.models import load_model
import os
import numpy as np
from gradcam import GradCAM
import cv2
from tqdm import tqdm
import pandas as pd

FLAGS = argHandler()
FLAGS.setDefaults()
GRADCAM_THRESH = 50
WHITE_THRESH = 85
WRITE_PATH = os.path.join(FLAGS.save_model_path, f"gradcam_thresh_{GRADCAM_THRESH}_white_thresh_{WHITE_THRESH}")
SAVED_PATH = os.path.join(FLAGS.save_model_path, f"saved_hl")

ANNOTATION_CSV_FILE = './data/radiology_segmentations.csv'

df = pd.read_csv(ANNOTATION_CSV_FILE)
try:
    os.makedirs(WRITE_PATH)
except:
    print("path already exists")

model_factory = ModelFactory()

if FLAGS.load_model_path != '' and FLAGS.load_model_path is not None:
    visual_model = load_model(FLAGS.load_model_path)
    if FLAGS.show_model_summary:
        visual_model.summary()
else:
    visual_model = model_factory.get_model(FLAGS)
FLAGS.batch_size = 1
generator = get_generator(FLAGS.test_csv, FLAGS)

images_names = generator.get_images_names()

iou_sum = 0
f1_sum = 0
images_count = 0
overlap_csv = {"image_name": [], "overlap_50": [], "overlap_60": [], "overlap_70": [], "iou": [], "f1": []}

for batch_i in tqdm(range(generator.steps)):
    batch, y = generator.__getitem__(batch_i)
    if y[0] == 0:
        continue
    predicted_class = y[0]
    label = FLAGS.classes[predicted_class]

    image_path = os.path.join(FLAGS.image_directory, images_names[batch_i])
    if os.path.isfile(os.path.join(SAVED_PATH, images_names[batch_i]) + '.npy'):
        heatmap = np.load(os.path.join(SAVED_PATH, images_names[batch_i]) + '.npy')
    else:
        preds = visual_model.predict(batch)
        cam = GradCAM(visual_model, predicted_class)
        heatmap = cam.compute_heatmap(batch)
    original = cv2.imread(image_path.replace('_224', ''))
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = apply_white_threshold(original, heatmap, WHITE_THRESH)
    oneshot = heatmap >= GRADCAM_THRESH
    oneshot = remove_corner_highlights(heatmap, oneshot)
    heatmap = oneshot * heatmap
    heatmap[heatmap > 0] = 200
    cam = GradCAM(visual_model, predicted_class)

    (heatmap, output) = cam.overlay_heatmap(heatmap, original, alpha=0.5)
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # evaluations
    masks = df[df['#filename'] == images_names[batch_i]]['region_shape_attributes']
    GT_mask = np.array(get_segmented_image(original, masks)) > 0
    if np.sum(GT_mask.astype(np.int)) == 0:
        continue
    GT_mask = apply_white_threshold(original, GT_mask, WHITE_THRESH)
    overlap_50 = is_covering_segmentation(GT_mask.astype(np.int), oneshot.astype(np.int), 0.5)
    overlap_60 = is_covering_segmentation(GT_mask.astype(np.int), oneshot.astype(np.int), 0.6)
    overlap_70 = is_covering_segmentation(GT_mask.astype(np.int), oneshot.astype(np.int), 0.7)

    iou = get_IOU(GT_mask.astype(np.int), oneshot.astype(np.int))
    f1 = get_f1(GT_mask.astype(np.int), oneshot.astype(np.int))
    f1_sum += f1
    iou_sum += iou
    blended = alpha_blend(np.array(output), GT_mask.astype(np.int))
    images_count += 1
    print(f"overall_overlap {f1_sum / images_count}")

    overlap_csv['image_name'].append(images_names[batch_i])
    overlap_csv['overlap_50'].append(overlap_50)
    overlap_csv['overlap_60'].append(overlap_60)
    overlap_csv['overlap_70'].append(overlap_70)
    overlap_csv['iou'].append(iou)
    overlap_csv['f1'].append(f1)

    cv2.imwrite(os.path.join(WRITE_PATH, images_names[batch_i]), blended)

print(f"overall_overlap {iou_sum / images_count}")

new_csv = pd.DataFrame(overlap_csv)

new_csv.to_csv(os.path.join(WRITE_PATH, 'overlap.csv'), index=False)
