import cv2
import numpy as np
import tkinter as tk
import os
import random
from tkinter import filedialog, messagebox
import glob
import re

def select_orientation_folder(original_path, current_directory):
    print(original_path)
    root = tk.Tk()
    root.withdraw()


    file_paths = filedialog.askdirectory(title="Select orientation folder")

    if not file_paths:
        raise ValueError("No folder selected.")

    files = sorted(glob.glob(os.path.join(file_paths, '*.png')), key=extract_number2)
    num_images = len(files)

    if num_images == 0:
        raise ValueError("No PNG images found in the selected folder.")
    return file_paths

    return file_paths

def extract_number2(file_name):
    base_name = os.path.basename(file_name)
    match = re.search(r'(\d+)', base_name)
    if match:
        return int(match.group(1))
    return float('inf')

def pseudo_imgs_generator(orientation_path, random_flag):

    files = sorted(glob.glob(os.path.join(orientation_path, '*.png')), key=extract_number2)
    num_images = len(files)

    pseudo_imgs = []
    [m1n, n1n, d] = np.shape(cv2.imread(files[0]))
    SizeIm = [n1n, m1n]

    if random_flag == 0:
        for i in range(3):
            pseudoimage = np.zeros((m1n, n1n, 3), dtype=np.uint8)
            if i == 0:
                pseudoimage[:,:,0] = cv2.imread(files[4], cv2.IMREAD_GRAYSCALE)
                pseudoimage[:,:,1] = cv2.imread(files[8], cv2.IMREAD_GRAYSCALE)
                pseudoimage[:, :, 2] = cv2.imread(files[12], cv2.IMREAD_GRAYSCALE)
            if i == 1:
                pseudoimage[:, :, 0] = cv2.imread(files[2], cv2.IMREAD_GRAYSCALE)
                pseudoimage[:, :, 1] = cv2.imread(files[6], cv2.IMREAD_GRAYSCALE)
                pseudoimage[:, :, 2] = cv2.imread(files[10], cv2.IMREAD_GRAYSCALE)
            if i == 2:
                pseudoimage[:, :, 0] = cv2.imread(files[8], cv2.IMREAD_GRAYSCALE)
                pseudoimage[:, :, 1] = cv2.imread(files[12], cv2.IMREAD_GRAYSCALE)
                pseudoimage[:, :, 2] = cv2.imread(files[16], cv2.IMREAD_GRAYSCALE)
            pseudo_imgs.append(pseudoimage)
    elif random_flag == 1:
        # Random selection
        for _ in range(3):
            pseudoimage = np.zeros((m1n, n1n, 3), dtype=np.uint8)
            selected_indices = random.sample(range(num_images), 3)
            pseudoimage[:, :, 0] = cv2.imread(files[selected_indices[0]], cv2.IMREAD_GRAYSCALE)
            pseudoimage[:, :, 1] = cv2.imread(files[selected_indices[1]], cv2.IMREAD_GRAYSCALE)
            pseudoimage[:, :, 2] = cv2.imread(files[selected_indices[2]], cv2.IMREAD_GRAYSCALE)
            pseudo_imgs.append(pseudoimage)

    return pseudo_imgs, SizeIm