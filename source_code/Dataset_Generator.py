# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:32:20 2024

@author: ezxtg6
"""
import numpy as np
import os
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops, find_contours
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from skimage.draw import polygon
import copy
from ultralytics import YOLO
import shutil
import random
from source_code.pseudoimage import apply_clahe, adjust_contrast, normalize_images
import math
import copy
# %%

def find_contour_final(binary_image,mode):
    
    inverted_image = ~binary_image
    if mode == 0:
        skeleton = np.uint8(skeletonize(inverted_image))
    elif mode == 1:
        skeleton = ~inverted_image
    else:
        skeleton = ~inverted_image
   # skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1)

    # Step 3: Replace the border pixels with 1s in the skeleton

    bool_ske_final = ~dilated_image.astype(bool)     
     
    inverted_image = ~binary_image
    if mode == 0 or mode == 1:
        skeleton = np.uint8(skeletonize(inverted_image))
    else:
        skeleton = ~inverted_image
   # skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(skeleton, kernel, iterations=1)
        skeleton = np.uint8(skeletonize(dilated_image))
        
    # Step 3: Replace the border pixels with 1s in the skeleton
    if mode == 0 or mode == 2:
        skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1) 
    
    bool_ske = ~dilated_image.astype(bool)

    if mode == 0:
        labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
    elif mode == 2:
        binary_imafe = np.pad(binary_image, pad_width=1, mode='constant', constant_values=1)
        bool_ske = binary_imafe.astype(bool)
        labeled_array_ske, num_features_ske = label(~bool_ske, return_num=True)
    elif mode == 3:
       int_image = (binary_image < 255).astype(int)
       skeleton = np.uint8(skeletonize(int_image))
       kernel = np.ones((3, 3), np.uint8)
       dilated_image = cv2.dilate(skeleton, kernel, iterations=1) 
       
       bool_ske = ~dilated_image.astype(bool)
       
       labeled_array_ske, num_features_ske = label(skeleton, return_num=True)
       
    else:
        labeled_array_ske, num_features_ske = label(~bool_ske, return_num=True)
    
    contours = find_contours(labeled_array_ske, 0.5)
    
    contour_image = np.zeros_like(binary_image)
    contour_points_list = []
    for contour in contours:
        contour_points = []
        for point in contour:
            y, x = point
            if mode == 0 or mode == 2:
                corrected_x = max(0, math.floor(x)-1) 
                corrected_y = max(0, math.floor(y)-1) 
                contour_points.append((corrected_x, corrected_y))
                
                contour_image[math.floor(y)-1, math.floor(x)-1] = 255
            else:

                corrected_x = max(0, round(x)) 
                corrected_y = max(0, round(y)) 
                contour_points.append((corrected_x, corrected_y))
                contour_image[round(y), round(x)] = 255
        contour_points_list.append(contour_points)
    pt = 1
    return contour_points_list   


