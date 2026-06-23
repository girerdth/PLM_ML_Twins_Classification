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
    """
    Extract contour point lists from a binary image, using a mode-dependent
    skeletonisation/labelling strategy.

    Parameters
    ----------
    binary_image : ndarray
        Input binary (or near-binary) image, where boundaries are typically
        marked by zero/low values against a background.
    mode : int
        Selects how the image is pre-processed before contour extraction:
        0 - Skeletonize the inverted image, dilate, pad the border with 1s,
            then re-dilate and label the *background* region (i.e. grain
            interiors) for contour tracing.
        1 - Use the inverted image directly as the "skeleton" (no
            skeletonize call) for the first pass; the second block also
            skeletonizes here, sharing mode 0's behaviour there.
        2 - Similar to 0, but skips the border padding in the first block,
            and in the final labelling step uses the original binary image
            (padded) rather than the dilated skeleton.
        3 - Treats binary_image as a grayscale-like array, thresholds it at
            255, skeletonizes the result, dilates, and labels the skeleton
            itself directly (not its background).

    Returns
    -------
    contour_points_list : list of list of tuple of int
        One list of (x, y) integer coordinates per contour found, with
        rounding/offset behaviour that differs by mode (see below).

    Purpose
    -------
    This function unifies several slightly different contour-extraction
    strategies (originally separate code paths) behind a single `mode`
    switch. All variants follow the same general idea: invert/skeletonize
    the input, dilate it slightly to close small gaps, label the resulting
    regions, and trace their boundaries with skimage's find_contours. Modes
    0 and 2 use a "floor and shift by -1" rounding convention (to compensate
    for the padding added in those modes), while modes 1 and 3 use plain
    rounding. Only the second `skeleton`/`bool_ske` recomputation feeds
    into the labelling step that is actually used.
    """

    # Invert the input so that boundaries/background become foreground for skeletonisation
    inverted_image = ~binary_image
    if mode == 0:
        # Reduce the inverted region to a 1-pixel-wide skeleton
        skeleton = np.uint8(skeletonize(inverted_image))
    elif mode == 1:
        # No skeletonisation here; just reuse the inverted image as-is
        skeleton = ~inverted_image
    else:
        skeleton = ~inverted_image

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1)

    # Step 3: Replace the border pixels with 1s in the skeleton
    # Recompute the skeleton from scratch (this is the version actually used downstream)
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


