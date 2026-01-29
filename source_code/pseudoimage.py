import cv2
import numpy as np
import tkinter as tk
import os
from tkinter import filedialog



def select_images():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select 3 images",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
    )
    if len(file_paths) != 3:
        print("Please select exactly 3 images.")
        return None
    return file_paths

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def adjust_contrast(image):
    min_val, max_val = np.min(image), np.max(image)
    return np.uint8((image - min_val) / (max_val - min_val) * 255)

def normalize_images(img1, img2, img3):
    stacked = np.stack([img1, img2, img3], axis=-1).astype(np.float32)
    min_val, max_val = np.min(stacked), np.max(stacked)
    normalized = (stacked - min_val) / (max_val - min_val) * 255
    return cv2.split(np.uint8(normalized))

def normalize_images_all(imgs):
    stacked = np.stack(imgs, axis=-1).astype(np.float32)
    min_val, max_val = np.min(stacked), np.max(stacked)
    normalized = (stacked - min_val) / (max_val - min_val) * 255
    return cv2.split(np.uint8(normalized))

def resize_images_to_match(img1, img2, img3):
    heights = [img1.shape[0], img2.shape[0], img3.shape[0]]
    widths = [img1.shape[1], img2.shape[1], img3.shape[1]]
    min_height = min(heights)
    min_width = min(widths)

    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))
    img3_resized = cv2.resize(img3, (min_width, min_height))

    return img1_resized, img2_resized, img3_resized

def measure_color_dominance(merged_img):
    b, g, r = cv2.split(merged_img)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    print(f"Mean intensity of Blue channel: {mean_b:.2f}")
    print(f"Mean intensity of Green channel: {mean_g:.2f}")
    print(f"Mean intensity of Red channel: {mean_r:.2f}")

    dominant_channel = np.argmax([mean_b, mean_g, mean_r])
    if dominant_channel == 0:
        print("Blue is the most dominant color.")
    elif dominant_channel == 1:
        print("Green is the most dominant color.")
    else:
        print("Red is the most dominant color.")

def merge_images(image_paths):
    img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(image_paths[2], cv2.IMREAD_GRAYSCALE)

    img1, img2, img3 = resize_images_to_match(img1, img2, img3)

    img1_clahe = apply_clahe(img1)
    img2_clahe = apply_clahe(img2)
    img3_clahe = apply_clahe(img3)

    img1_contrast = adjust_contrast(img1_clahe)
    img2_contrast = adjust_contrast(img2_clahe)
    img3_contrast = adjust_contrast(img3_clahe)

    img1_norm, img2_norm, img3_norm = normalize_images(img1_contrast, img2_contrast, img3_contrast)

    # Merge into an RGB image (OpenCV uses BGR format)
    merged_img = cv2.merge([img3_norm, img2_norm, img1_norm])

    # Print which image corresponds to which channel
    print(f"Image 1 ({image_paths[0]}) is assigned to the Red channel.")
    print(f"Image 2 ({image_paths[1]}) is assigned to the Green channel.")
    print(f"Image 3 ({image_paths[2]}) is assigned to the Blue channel.")

    return merged_img

def main():
    image_paths = select_images()
    if not image_paths:
        return None

    merged_img = merge_images(image_paths)

    if merged_img is not None:
        cv2.imshow('Merged Image', merged_img)
        cv2.waitKey(0)

        measure_color_dominance(merged_img)

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if save_path:
            cv2.imwrite(save_path, merged_img)
            print(f"Image saved as {save_path}")

            # Extract the file name from the save path
            file_name = os.path.basename(save_path)
            return save_path, image_paths[0]

    return None

if __name__ == "__main__":
    main()
