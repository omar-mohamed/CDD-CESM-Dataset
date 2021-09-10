from __future__ import absolute_import, division

from visual_model_selector import ModelFactory
from configs import argHandler  # Import the default arguments
from model_utils import set_gpu_usage, get_generator
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw
import os
import numpy as np
from gradcam import GradCAM
import cv2
from tqdm import tqdm
import pandas as pd
import json
from scipy import ndimage
import matplotlib.pyplot as plt

FLAGS = argHandler()
FLAGS.setDefaults()
GREATER = 50
WHITE = 85
WRITE_PATH = os.path.join(FLAGS.save_model_path, f"greater_{GREATER}_white_{WHITE}_P7_L_DM_MLO")
SAVED_PATH = os.path.join(FLAGS.save_model_path, f"saved_hl")

ANNOTATION_CSV_FILE = './data/radiology_segmentations.csv'


def get_polygon_formatted(x_points, y_points):
    points = []
    for i in range(len(x_points)):
        points.append((x_points[i], y_points[i]))
    return points


def get_segmented_image(image, masks):
    img_mask = Image.new('L', (image.shape[1], image.shape[0]), 0)
    for mask in masks:
        if mask == '{}':
            continue
        mask = json.loads(mask)
        if mask['name'] == 'polygon':
            poly = get_polygon_formatted(mask['all_points_x'], mask['all_points_y'])
            ImageDraw.Draw(img_mask).polygon(poly, outline=1, fill=1)
        elif mask['name'] == 'ellipse' or mask['name'] == 'circle' or mask['name'] == 'point':
            if mask['name'] == 'circle':
                mask['rx'] = mask['ry'] = mask['r']
            elif mask['name'] == 'point':
                mask['rx'] = mask['ry'] = 25
            ellipse = [(mask['cx'] - mask['rx'], mask['cy'] - mask['ry']),
                       (mask['cx'] + mask['rx'], mask['cy'] + mask['ry'])]
            ImageDraw.Draw(img_mask).ellipse(ellipse, outline=1, fill=1)
        # elif mask['name'] == 'point':
        #     img_mask[mask['cx']]
    return img_mask


def alpha_blend(img, mask):
    ALPHA = 0.5
    print(mask.shape)
    mask *= (ALPHA * mask * 255).astype(np.uint8)
    # mask = np.dstack([mask]*3)
    redImg = np.zeros(img.shape, np.uint8)
    redImg[:, :] = (0, 0, 255)
    print(mask.shape)
    print(redImg.shape)

    redMask = cv2.bitwise_and(redImg.astype(np.uint8), redImg.astype(np.uint8), mask=mask.astype(np.uint8))
    blended = cv2.addWeighted(redMask, ALPHA, img, 1, 0, img)
    blended = img.astype(np.uint16) + redMask  # np.expand_dims(mask, axis=-1)
    blended = blended.clip(0, 255)
    return blended.astype(np.uint8)


def remove_corner_highlights(heatmap, oneshot):
    oneshot_85 = heatmap >= np.percentile(heatmap, 85)
    labeled, nr_objects = ndimage.label(oneshot_85.astype(np.int))
    if nr_objects > 1 and labeled[0, 0] > 0:
        oneshot[labeled == labeled[0, 0]] = 0
    return oneshot


def get_overlap_percentage(gt, seg):
    return np.sum(np.logical_and(gt, seg)) / np.sum(np.logical_or(gt, seg))


def get_IOU(gt, seg):
    return (get_overlap_percentage(gt, seg) + get_overlap_percentage(np.logical_not(gt).astype(int),
                                                                     np.logical_not(seg).astype(int))) / 2


def calc_f1(gt, seg):
    return 2 * np.sum(np.logical_and(gt, seg)) / np.sum(gt + seg)


def get_f1(gt, seg):
    return (calc_f1(gt, seg) + calc_f1(np.logical_not(gt).astype(int), np.logical_not(seg).astype(int))) / 2


def is_covering_segmentation(gt, seg, percentage=0.5):
    return int((np.sum(np.logical_and(gt, seg)) / np.sum(gt)) >= percentage)


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # plt.imshow(gray)
    # plt.show()
    return gray


def apply_white_threshold(o, h):
    o_gray = rgb2gray(o)
    o_60 = o_gray < np.percentile(o_gray, WHITE)
    h[o_60] = 0
    return h


df = pd.read_csv(ANNOTATION_CSV_FILE)
try:
    os.makedirs(WRITE_PATH)
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
test_generator = get_generator(FLAGS.train_csv, FLAGS)

images_names = test_generator.get_images_names()

iou_sum = 0
f1_sum = 0
# overlap_70_sum=0
images_count = 0
overlap_csv = {"image_name": [], "overlap_50": [], "overlap_60": [], "overlap_70": [], "iou": [], "f1": []}

for batch_i in tqdm(range(test_generator.steps)):
    # if images_names[batch_i] != 'P7_L_DM_MLO.jpg':
    #   continue
    batch, y = test_generator.__getitem__(batch_i)
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
    heatmap = apply_white_threshold(original, heatmap)
    oneshot = heatmap >= GREATER
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
    GT_mask = apply_white_threshold(original, GT_mask)
    overlap_50 = is_covering_segmentation(GT_mask.astype(np.int), oneshot.astype(np.int), 0.5)
    overlap_60 = is_covering_segmentation(GT_mask.astype(np.int), oneshot.astype(np.int), 0.6)
    overlap_70 = is_covering_segmentation(GT_mask.astype(np.int), oneshot.astype(np.int), 0.7)

    # overlap_70_sum+=overlap_70
    iou = get_IOU(GT_mask.astype(np.int), oneshot.astype(np.int))
    f1 = get_f1(GT_mask.astype(np.int), oneshot.astype(np.int))
    f1_sum += f1
    iou_sum += iou
    blended = alpha_blend(np.array(original), GT_mask.astype(np.int))
    # blended = np.array(output)
    images_count += 1
    print(f"overall_overlap {f1_sum / images_count}")

    overlap_csv['image_name'].append(images_names[batch_i])
    overlap_csv['overlap_50'].append(overlap_50)
    overlap_csv['overlap_60'].append(overlap_60)
    overlap_csv['overlap_70'].append(overlap_70)
    overlap_csv['iou'].append(iou)
    overlap_csv['f1'].append(f1)

    # print(f"overall_overlap {overlap_sum / images_count}")
    # print(f"overall_overlap {overlap_70_sum / images_count}")
    cv2.imwrite(os.path.join(WRITE_PATH, images_names[batch_i]), blended)

print(f"overall_overlap {iou_sum / images_count}")

new_csv = pd.DataFrame(overlap_csv)

new_csv.to_csv(os.path.join(WRITE_PATH, 'overlap.csv'), index=False)
