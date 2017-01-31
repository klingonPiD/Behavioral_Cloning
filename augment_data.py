import pandas as pd
from scipy.misc import imread, imresize
import numpy as np
import cv2
import math
from skimage import transform

def apply_random_augmentation(row_data, valid_flag=False):
    """Reads images from files and apply random
    augmentation to it. Modifies steering
    angles accordingly """
    st_angle = row_data['steering']
    # 1. Choose left, right or center cam image and adjust st angle accordingly
    cam_num = np.random.choice([0,1,2]) #left, center, right cams
    # for beta simulator, left and right cam do not exist. Set cam to center
    if pd.isnull(row_data['left']) or pd.isnull(row_data['right']):
        cam_num = 1
    # Angle shift required to adjust left-right images to center images
    angle_shift = 0.23
    if cam_num == 0:
        filename = row_data['left'].strip()
        st_angle += angle_shift
    elif cam_num == 1:
        filename = row_data['center'].strip()
    else:
        filename = row_data['right'].strip()
        st_angle -= angle_shift
    # Read the image
    im = imread(filename)
    # for validation - by pass this augmentation step
    if not valid_flag:
        # 2. do random left right translations
        trans = np.random.randint(-50, 50) #-75 to 75
        tform_aug = transform.AffineTransform(translation=(trans, 0))
        for ch in range(3):
            im[:, :, ch] = fast_warp(im[:, :, ch], tform_aug, output_shape=(im.shape[0], im.shape[1]), mode='constant')
        st_angle += trans * 0.005
    # 3. Crop and resize the image
    # crop (top 1/4 and bottom 1/5) and resize image
    lx, ly, lz = im.shape
    im = im[math.floor(lx / 4): math.floor(-lx / 5), :, :]
    im = imresize(im, (64, 64, 3))  # im = imresize(im, (66, 200, 3))
    #4. Convert RGB images to HSV
    im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # Add some random brightness by scaling the V channel
    #im_hsv[:,:,2] = im_hsv[:,:,2] * np.random.uniform(0.25, 1.0)
    im = im_hsv
    #5. Do fliplr to preserve negative and positive balance of steering angle data
    flip = np.random.choice([0,1])
    if flip:
        im = cv2.flip(im,1) #np.fliplr(im)
        st_angle = -st_angle
    return im, st_angle

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params,
                                          output_shape=output_shape, mode=mode)
