# -*- coding: utf-8 -*-
"""
Created on Thu May 15 08:41:03 2025

@author: ezxtg6
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.morphology import disk, dilation
from scipy.ndimage import distance_transform_edt
import cv2

def fill_missing_nearest(image):
    """Fill missing (NaN) pixels using nearest neighbor interpolation."""
    mask = np.isnan(image)
    filled = image.copy()
    dist, inds = distance_transform_edt(mask, return_indices=True)
    filled[mask] = image[tuple(inds[:, mask])]
    return filled

def grain_orientation(grainstats, mode, FinalGB, colormap_path=None):
    width, height = FinalGB.shape
    X1, Y1 = np.meshgrid(np.arange(width), np.arange(height))
    FinalGB = ~FinalGB.astype(bool)

    if mode == 3:
        if colormap_path is None:
            raise ValueError("Colormap path must be provided for mode 3.")
        
        cropped_A = imread(colormap_path)
        A = cropped_A[1:, 1:-1, :]
        pixelX = A.shape[1] / 180
        pixelY = A.shape[0] / 90

        all_colors = []

        for grain in grainstats:
            if grain.ID == 51:
                toto = 1
            pixels = np.array(grain.PixelList)
            inclination = min(grain.Inclination, 89)
            azimuth = grain.Azimuth
            x, y = inclination * np.cos(np.radians(azimuth)), inclination * np.sin(np.radians(azimuth))
            X = int(round(x * pixelX + A.shape[1] / 2))
            Y = int(round(-y * pixelY + A.shape[0]))
            X = np.clip(X, 0, A.shape[1] - 1)
            Y = np.clip(Y, 0, A.shape[0] - 1)
            color = A[Y, X]  # RGB triplet

            colored_pixels = np.hstack((pixels, np.tile(color, (len(pixels), 1))))
            all_colors.append(colored_pixels)

        all_colors = np.vstack(all_colors)

        Zfinal2 = np.zeros((width, height, 3), dtype=np.float32)
        x_idx = all_colors[:, 1].astype(int)
        y_idx = all_colors[:, 0].astype(int)

        for i in range(3):  # RGB channels
            Z = np.zeros((width, height), dtype=np.float32)
            Z[y_idx, x_idx] = all_colors[:, 2 + i]
            Z[Z == 0] = np.nan
            Z = fill_missing_nearest(Z)
            Zfinal2[:, :, i] = Z

        ColorMap = Zfinal2.astype(np.uint8)

    elif mode == 1:
        all_colors = []

        for grain in grainstats:
            pixels = np.array(grain.PixelList)
            azimuth = grain.Azimuth if grain.Azimuth != 0 else 0.2
            colored_pixels = np.hstack((pixels, np.full((len(pixels), 1), azimuth)))
            all_colors.append(colored_pixels)

        all_colors = np.vstack(all_colors)

        Zfinal2 = np.zeros((height, width), dtype=np.float32)
        x_idx = all_colors[:, 1].astype(int)
        y_idx = all_colors[:, 0].astype(int)
        Zfinal2[y_idx, x_idx] = all_colors[:, 2]

        Zfinal2[Zfinal2 == 0] = np.nan
        Zfinal2 = fill_missing_nearest(Zfinal2)
        ColorMap = Zfinal2

    else:  # mode == 2 or any other (assume inclination map)
        all_colors = []

        for grain in grainstats:
            pixels = np.array(grain.PixelList)
            inclination = grain.Inclination if grain.Inclination != 0 else 0.1
            colored_pixels = np.hstack((pixels, np.full((len(pixels), 1), inclination)))
            all_colors.append(colored_pixels)

        all_colors = np.vstack(all_colors)

        Zfinal2 = np.zeros((height, width), dtype=np.float32)
        x_idx = all_colors[:, 1].astype(int)
        y_idx = all_colors[:, 0].astype(int)
        Zfinal2[y_idx, x_idx] = all_colors[:, 2]

        Zfinal2[Zfinal2 == 0] = np.nan
        Zfinal2 = fill_missing_nearest(Zfinal2)
        ColorMap = Zfinal2

    return ColorMap