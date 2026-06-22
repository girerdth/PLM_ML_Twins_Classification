import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
import os
import glob
import re
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_FOLDER = "Analysis_Results"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def extract_number2(file_name):
    base_name = os.path.basename(file_name)
    match = re.search(r'(\d+)', base_name)
    if match:
        return int(match.group(1))
    return float('inf')


def select_orientation_folder():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(title="Select folder")

    if not folder_path:
        return None  # Return None so the "While" loop can exit gracefully

    # Search for .bmp files specifically
    files = sorted(glob.glob(os.path.join(folder_path, '*.bmp')), key=extract_number2)

    if len(files) == 0:
        # Also check for .png if .bmp isn't there
        files = sorted(glob.glob(os.path.join(folder_path, '*.png')), key=extract_number2)

    if len(files) == 0:
        raise ValueError("No images found in the selected folder.")

    return files  # Return the LIST of file paths, not the folder string



def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def adjust_contrast(image):
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val == 0: return image
    return np.uint8((image - min_val) / (max_val - min_val) * 255)


def normalize_images(img1, img2, img3):
    stacked = np.stack([img1, img2, img3], axis=-1).astype(np.float32)
    min_val, max_val = np.min(stacked), np.max(stacked)
    normalized = (stacked - min_val) / (max_val - min_val) * 255
    return cv2.split(np.uint8(normalized))


def resize_images_to_match(img1, img2, img3):
    heights = [img1.shape[0], img2.shape[0], img3.shape[0]]
    widths = [img1.shape[1], img2.shape[1], img3.shape[1]]
    min_height, min_width = min(heights), min(widths)

    return (cv2.resize(img1, (min_width, min_height)),
            cv2.resize(img2, (min_width, min_height)),
            cv2.resize(img3, (min_width, min_height)))


def save_intensity_boxplot(merged_img, base_name):
    b, g, r = cv2.split(merged_img)
    df = pd.DataFrame({
        'Red': r.flatten(),
        'Green': g.flatten(),
        'Blue': b.flatten()
    })

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 6))
    # 4. Create Plot
    sns.boxplot(
        data=df,
        palette=['red', 'green', 'blue'],
        width=0.5,
        showfliers=False,
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black'),
        ax=ax
    )

    # 5. Axis and Spine Customization
    plt.ylabel('Intensity Value', fontsize=14)
    plt.xlabel('Color Channel', fontsize=14)
    plt.ylim(0, 255)  # Standard RGB range is 0-255; change back to 150 if preferred

    # Ensure all borders (spines) are visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # Configure ticks to point outward
    ax.tick_params(axis='both', which='major', direction='out',
                   color='black', length=6, width=1.0,
                   bottom=True, left=True)


    plot_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_boxplot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close plot to free memory in loop
    return plot_path


def process_batch(image_paths):
    # Load as Grayscale

    merged_imgs = []
    avgs = []


    gap = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for image in image_paths:
        img1 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img1_norm = adjust_contrast(apply_clahe(img1))
        avg = cv2.mean(img1_norm)[0]
        avgs.append(avg)
    img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    img1_norm = adjust_contrast(apply_clahe(img1))
    avg = cv2.mean(img1_norm)[0]
    avgs.append(avg)

    for i in gap:

        img1 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        if i == 9:
            img3 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        else:
            img3 = cv2.imread(image_paths[2*i], cv2.IMREAD_GRAYSCALE)


        img1, img2, img3 = resize_images_to_match(img1, img2, img3)

        # Processing pipeline
        img1_norm, img2_norm, img3_norm = normalize_images(
            adjust_contrast(apply_clahe(img1)),
            adjust_contrast(apply_clahe(img2)),
            adjust_contrast(apply_clahe(img3))
        )

        merged_img = cv2.merge([img3_norm, img2_norm, img1_norm])

        merged_imgs.append(merged_img)





    return merged_imgs, avgs


def main():
    print("Starting continuous image processor...")
    print(f"All files will be saved in: {os.path.abspath(OUTPUT_FOLDER)}")

    image_paths = select_orientation_folder()
    print("HERE")
    # Generate a unique timestamp for this set


    try:
        merged_imgs, avgs = process_batch(image_paths)

        # Save Merged Image
        for m in range(len(merged_imgs)):
            img_save_path = os.path.join(OUTPUT_FOLDER, f"merged_{m}.png")
            cv2.imwrite(img_save_path, merged_imgs[m])
            # Generate and Save Boxplot
            #plot_path = save_intensity_boxplot(merged_imgs[m], f"analysis_{m}")

        x = np.arange(0,190,10)
        y = np.array(avgs)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y,  linestyle='-', color='black')  # Added marker for clarity
        plt.xlabel('Angle')
        plt.ylabel('Average Intensity')
        plt.xlim(0, 180)
        plt.ylim(50,150)
        plt.yticks([50,75,100,125,150])
        plt.grid(False)

        # Define the save path
        save_path = os.path.join(OUTPUT_FOLDER, "prout.png")

        # Use bbox_inches='tight' to ensure labels aren't cut off
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    except Exception as e:
        print(f"Error processing batch: {e}")


if __name__ == "__main__":
    main()