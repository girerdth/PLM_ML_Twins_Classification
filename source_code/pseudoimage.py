import cv2
import numpy as np
import tkinter as tk
import os
from tkinter import filedialog
import matplotlib.pyplot as plt
import csv # Add this at the top of your script
import pandas as pd
import seaborn as sns

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


def plot_color_dominance(mean_b, mean_g, mean_r):
    channels = ['Blue', 'Green', 'Red']
    intensities = [mean_b, mean_g, mean_r]
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(channels, intensities, color=colors, edgecolor='black', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Color Channels')
    plt.ylabel('Mean Intensity (0-255)')
    plt.title('Mean Color Intensity Distribution')
    plt.ylim(0, 255)  # Since pixel values are 0-255

    # Add the numerical value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, round(yval, 2), ha='center', va='bottom')

    plt.show()

def measure_color_dominance(merged_img, save_path=None):
    b, g, r = cv2.split(merged_img)

    # Calculate means
    data = {
        "Channel": ["Red", "Green", "Blue"],
        "Mean_Intensity": [np.mean(r), np.mean(g), np.mean(b)]
    }

    # Print for manual copy-pasting to Excel
    print("\n--- EXCEL DATA ---")
    print("Channel, Mean_Intensity")
    for i in range(3):
        print(f"{data['Channel'][i]}, {data['Mean_Intensity'][i]:.4f}")
    print("------------------\n")

    # Automatically save to CSV if a path is provided
    if save_path:
        csv_path = os.path.splitext(save_path)[0] + "_stats.csv"
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Channel", "Mean Intensity"])
            for i in range(3):
                writer.writerow([data['Channel'][i], data['Mean_Intensity'][i]])
        print(f"Stats saved for Excel at: {csv_path}")

    return data


def plot_intensity_boxplot(merged_img, filename="intensity_plot.png"):
    # 1. Prepare Data
    b, g, r = cv2.split(merged_img)
    df = pd.DataFrame({
        'Red': r.flatten(),
        'Green': g.flatten(),
        'Blue': b.flatten()
    })

    # 2. Set Global Styles (Arial and No Grid)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_style("white") # "white" style removes gridlines by default

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Create Plot
    sns.boxplot(
        data=df,
        palette=['red', 'green', 'blue'],
        width=0.5,
        showfliers=False,
        # Set box outlines to black
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black'),
        ax=ax
    )

    # 4. Axis and Tick Customization
    plt.ylabel('Intensity Value', fontsize=14)
    plt.xlabel('Color Channel', fontsize=14)
    plt.ylim(0, 150)

    # Force black spines (the border lines)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_edgecolor('black')
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_visible(True)

    # CRITICAL: Explicitly show the ticks and point them out
    ax.tick_params(axis='both', which='major', direction='out',
                   color='black', length=6, width=1.0,
                   bottom=True, left=True)  # Ensures they are turned ON

    # Optional: ensure ticks only appear where labels are
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Force black spines and outside ticks
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_visible(True)

    ax.tick_params(direction='out', color='black', length=6)

    # 5. Save and Show
    plt.tight_layout()
    plt.savefig(filename, dpi=300, format='png')
    print(f"Image saved as {filename}")
    plt.show()

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


def export_all_pixels_to_csv(merged_img, save_path):
    # 1. Split the BGR image (OpenCV format)
    b, g, r = cv2.split(merged_img)

    # 2. Flatten the 2D arrays into 1D lists (all pixels)
    r_flat = r.flatten()
    g_flat = g.flatten()
    b_flat = b.flatten()

    csv_path = os.path.splitext(save_path)[0] + "_all_pixels.csv"

    # Check if data exceeds Excel's row limit (~1 million rows)
    if len(r_flat) > 1048576:
        print(f"Warning: Image has {len(r_flat)} pixels, which exceeds Excel's row limit.")
        print("Exporting anyway, but you may need to use Power Pivot or Python to analyze.")

    # 3. Save using pandas (much faster for large datasets)
    df = pd.DataFrame({
        'Red': r_flat,
        'Green': g_flat,
        'Blue': b_flat
    })

    df.to_csv(csv_path, index=False)
    print(f"All pixel values saved to: {csv_path}")


def main():
    image_paths = select_images()
    if not image_paths:
        return None

    merged_img = merge_images(image_paths)

    if merged_img is not None:
        #cv2.imshow('Merged Image', merged_img)
        #cv2.waitKey(0)

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        #plot_intensity_boxplot(merged_img)
        if save_path:
            cv2.imwrite(save_path, merged_img)
            print(f"Image saved as {save_path}")

            # Extract the file name from the save path
            file_name = os.path.basename(save_path)
            return save_path, image_paths[0]

    return None

if __name__ == "__main__":
    main()
