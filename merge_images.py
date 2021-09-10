import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

write_path = './data/images_hybrid'
csv_name = 'all_data_hybrid.csv'
dataset_df = pd.read_csv('./data/all_data_3c.csv')

try:
    os.makedirs(write_path)
except:
    pass

def plt_image(img):
    plt.imshow(img)
    plt.show()

def crop(img, margins):
    return img[margins[0]:margins[1],margins[2]:margins[3]]

def crop_image(img,tol=0,margin=100, is_MLO = False):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    if is_MLO:
        row_start += int(0.3*row_end)
    row_start = max(0,row_start-margin)
    col_start = max(0,col_start-margin)
    return img[row_start:row_end+margin,col_start:col_end+margin],(row_start,row_end+margin,col_start,col_end+margin)


def make_dict(dataset_df):
    dict={}
    for column in dataset_df:
        dict[column] = []
    return dict

def add_row(dict,df_row):
    for key in dict.keys():
        dict[key].append(df_row[key])

def combine(dm,cm):
    dm = dm / 255

    hybrid = cm * dm
    # print(hybrid.max(), hybrid.min())

    hybrid = hybrid / hybrid.max()
    hybrid = (hybrid * 255).astype(np.uint8)
    return hybrid
    # plt_image(hybrid)

new_csv = make_dict(dataset_df)

pbar = tqdm(total=len(dataset_df))

for i, row in dataset_df.iterrows():
    image_name = row['Image_name']
    image_name += '.jpg'
    if 'CM' in image_name:
        try:
            cm = cv2.imread(f"./data/images/{image_name}")
            dm_name = image_name.replace("CM","DM")
            dm = cv2.imread(f"./data/images/{dm_name}")
        except:
            print(f"Did no find dm for {image_name}")
            continue

        cm, margins = crop_image(cm, is_MLO=('MLO' in image_name))
        dm = crop(dm, margins)

        hybrid = combine(dm,cm)
        # resize if need be

        hybrid = Image.fromarray(hybrid)

        hybrid_name = image_name.replace("CM","H")

        row['Image_name'] = hybrid_name
        row['Type'] = 'H'
        add_row(new_csv,row)

        hybrid.save(write_path + f"/{hybrid_name}")
    pbar.update(1)
pbar.close()

new_csv=pd.DataFrame(new_csv)

new_csv.to_csv(os.path.join("./data",csv_name), index=False)

