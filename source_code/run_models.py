# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:38:52 2025

@author: Thomas Girerd
"""
# %% Python packages

# %% Python packages
import numpy as np
import os
import cv2
import glob
import re
import time
import math
import shutil
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
from skimage import io, img_as_ubyte
from skimage.filters.rank import modal
from skimage.measure import find_contours
from skimage.morphology import square
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import make_valid
from shapely.errors import ShapelyError
import torch
from concurrent.futures import ThreadPoolExecutor

# %% Own scripts
from source_code.Dataset_Generator import find_contour_final
import source_code.Grain_functions as Grain_functions
from source_code.amplify_methods import select_orientation_folder, pseudo_imgs_generator, orientation_and_classification


# %% Functions

# Define a function to extract the numerical part of the file name
def extract_number(file_name):
    match = re.search(r'(\d+)_Input', file_name)
    if match:
        return int(match.group(1))
    return float('inf')  # Return infinity if no number is found

def extract_colour(twins_red):
    twins_red = cv2.cvtColor(twins_red, cv2.COLOR_BGR2RGB)
    twins_red[twins_red < 255] = 0

    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]

    twins_red[np.all(twins_red == black, axis=-1)] = white

    twins_red = cv2.cvtColor(twins_red, cv2.COLOR_RGB2GRAY)

    return twins_red

def poly_line(gr):
    pts = gr.ContourPoints

    # 1) Build the raw geometry
    if len(pts) >= 3:
        geom = Polygon(pts)
    elif len(pts) == 2:
        geom = LineString(pts)
    elif len(pts) == 1:
        geom = Point(pts[0])
    else:
        # no points → return an empty geometry
        return Point()

        # 2) Try to make it valid
    try:
        # preferred in Shapely 2.x
        geom = make_valid(geom)
    except (ImportError, AttributeError, ShapelyError):
        # fallback for older Shapely versions or any failure
        if not geom.is_valid:
            geom = geom.buffer(0)

    return geom

def get_file_name_without_extension(file_path):
    """
    Extracts the file name without the extension from a file path.

    Args:
        file_path (str): The full path or name of the file.

    Returns:
        str: The file name without the extension.
    """
    file_name = os.path.basename(file_path)
    file_name_without_extension, _ = os.path.splitext(file_name)
    return file_name_without_extension

def simple_filtered(image_path):
    """
    Applies a modal filter to the input image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Filtered image.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray = img_as_ubyte(gray)
    size = 5
    footprint = square(size)
    filtered = modal(gray, footprint)
    return filtered

def advanced_filtered(image):
    """
    Applies an advanced filter to highlight specific regions in red.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Filtered image with highlighted regions.
    """
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    grey_mask = (image > 0) & (image < 255)
    color[grey_mask] = (0, 0, 255)
    return color

def clear_directory(directory):
    """Remove all files and subdirectories in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def delete_directory(directory):
    """Delete the specified directory and all its contents."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        print(f"Directory does not exist: {directory}")


def predict_twins_fast(images, model_twins):
    h, w = images[0].shape[:2]
    # We start with a blank canvas
    final_twins_mask = np.zeros((h, w), dtype=np.uint8)

    # Batch Predict (save=False is key)
    results = model_twins.predict(images, conf=0.5, imgsz=640, save=False, verbose=False)

    all_twins_contours = []

    for r in results:
        # Check if any masks were actually found
        if r.masks is not None:
            # r.masks.data contains the binary masks on the GPU/CPU
            # We scale them to the original image size
            masks = r.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                # Resize mask to original image size if necessary
                mask_resized = cv2.resize(mask, (w, h))

                # Add this mask to our final combined mask
                final_twins_mask[mask_resized > 0] = 255

                # Keep the contour points for your other functions
                if r.masks.xy is not None:
                    all_twins_contours.append(r.masks.xy[i])

    # Your logic: twins are black (0), background is white (255)
    # So we flip the mask
    final_twins = cv2.bitwise_not(final_twins_mask)

    return final_twins, all_twins_contours

def predict_grains(image, grains_model, predict_base_dir):
    """
    Predicts grains using a YOLO model.

    Args:
        image_path (str): Path to the input image.
        grains_model (YOLO): YOLO model for grains.
        predict_base_dir (str): Base directory for prediction results.

    Returns:
        str: Directory containing the prediction results for grains.
    """
    model_grains = YOLO(grains_model)

    grains_predict_dir = os.path.join(predict_base_dir, 'grains')
    delete_directory(grains_predict_dir)

    model_grains.predict(
        image,
        save=True,
        save_txt=True,
        conf=0.5,
        imgsz=640,
        max_det=3000,
        save_conf=True,
        project=predict_base_dir,
        name='grains'
    )

    return grains_predict_dir

def predict_grains_and_twins(image_path, grains_model, twins_model, predict_base_dir):
    """
    Predicts grains and twins using YOLO models in parallel, saving results in separate directories.

    Args:
        image_path (str): Path to the input image.
        grains_model (YOLO): YOLO model for grains.
        twins_model (YOLO): YOLO model for twins.
        predict_base_dir (str): Base directory for prediction results.

    Returns:
        tuple: (grains_dir, twins_dir) where `grains_dir` and `twins_dir` are the directories containing the prediction results.
    """
    # Clear the predict_base_dir to ensure it's empty
    clear_directory(predict_base_dir)

    model_grains = YOLO(grains_model)
    model_ML = YOLO(twins_model)


    with ThreadPoolExecutor() as executor:
        # Use unique project names for grains and twins
        executor.submit(
            model_grains.predict,
            image_path,
            save=True,
            save_txt=True,
            conf=0.5,
            imgsz=640,
            max_det=3000,
            save_conf=True,
            project=predict_base_dir,
            name='grains'
        )
        executor.submit(
            model_ML.predict,
            image_path,
            save=True,
            save_txt=True,
            conf=0.5,
            imgsz=640,
            max_det=3000,
            save_conf=True,
            project=predict_base_dir,
            name='twins'
        )

    # Get the latest prediction directories
    grains_dir = os.path.join(predict_base_dir, 'grains')
    twins_dir = os.path.join(predict_base_dir, 'twins')

    return grains_dir, twins_dir

def create_directories(final_folder):
    """
    Create necessary directories for results.

    Args:
        final_folder (str): Path to the final folder.
    """
    os.makedirs(final_folder, exist_ok=True)
    label_folder = os.path.join(final_folder, "labels")
    os.makedirs(label_folder, exist_ok=True)
    mask_folder = os.path.join(final_folder, "masks")
    os.makedirs(mask_folder, exist_ok=True)
    for insides in ['Grains', 'Twins', 'Both']:
        os.makedirs(os.path.join(mask_folder, insides), exist_ok=True)

def process_grains(grains_dir, image_path, SizeIm):
    """
    Process grain contours and generate grain matrix.

    Args:
        grains_dir (str): Directory containing YOLO prediction results for grains.
        image_path (str): Path to the input image.
        SizeIm (list): Size of the image as [width, height].

    Returns:
        tuple: (contour_points_grains, confidences, contour_mask)
    """
    # Use pathlib for cleaner path handling
    grains_labels_predict = Path(grains_dir) / 'labels'
    files = list(grains_labels_predict.glob('*.txt'))

    if not files:
        raise FileNotFoundError(f"No label files found in {grains_labels_predict}")

    # Determine the correct label file name
    if len(files) >= 1:
        if files[0].stem == "image0":
            label_name = 'image0.txt'
        else:
            label_name = Path(image_path).stem + '.txt'
    else:
        label_name = Path(image_path).stem + '.txt'

    label_file = grains_labels_predict / label_name

    # Read contours and confidences
    segmented_contours, confidences = Grain_functions.read_contours(str(label_file), SizeIm, 0)

    # Pre-allocate Seg_Grain as a list
    Seg_Grain = []
    centers = []
    contours_array = []

    for contour in segmented_contours:
        contour_array = np.array(contour, dtype=np.int32)
        points_inside = Grain_functions.get_integer_points_inside_contour(contour_array)
        points = np.array(points_inside, dtype=np.int32)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        centers.append((center_x, center_y))
        contours_array.append(contour_array)
        Seg_Grain.append(Grain_functions.Grain(points, contour_array, (center_x, center_y), len(points), confidences[len(Seg_Grain)], len(Seg_Grain) + 1))

    # Sort grains by size
    Seg_Grain = sorted(Seg_Grain, key=lambda x: x.size, reverse=True)

    # Handle overlapping grains

    overlapping_grains, ID_grains = Grain_functions.find_overlapping_grains(Seg_Grain)


    Grain_functions.handle_overlapping_grains(overlapping_grains, SizeIm)

    # Generate grain matrix and mask
    grain_matrix, mask = Grain_functions.generate_grain_matrix(Seg_Grain, SizeIm)
    grain_matrix2 = Grain_functions.fill_contour_gaps(Seg_Grain, grain_matrix, mask, SizeIm)
    grain_matrix2 = Grain_functions.add_missing_regions(grain_matrix2, Seg_Grain, SizeIm)
    contour_mask = Grain_functions.create_contour_mask(grain_matrix2)
    contour_points_grains = find_contour_final(contour_mask, 2)
    contour_mask = contour_mask * 255
    contour_mask = np.rot90(contour_mask)
    contour_mask = np.flipud(contour_mask)

    return contour_points_grains, confidences, contour_mask


def predict_and_process_grains(image, model_grains, SizeIm):
    """
    Predicts and processes grains entirely in memory.
    """
    # 1. Predict in memory (save=False is much faster)
    # Using [image] as a list even for one image keeps logic consistent
    results = model_grains.predict(image, conf=0.5, imgsz=640, max_det=3000, save=False, verbose=False)

    # 2. Extract data directly from the result object
    r = results[0]
    # If your model is a Segmentation model, use r.masks.xy
    # If it is a Detection model, use r.boxes.xywh
    raw_contours = r.masks.xy if r.masks is not None else []
    confidences = r.boxes.conf.cpu().numpy()

    Seg_Grain = []
    for i, contour in enumerate(raw_contours):
        contour_array = np.array(contour, dtype=np.int32)

        # Calculate pixels inside (Keep this if it's a custom logic you need)
        points_inside = Grain_functions.get_integer_points_inside_contour(contour_array)
        points = np.array(points_inside, dtype=np.int32)

        if len(points) == 0: continue  # Skip empty detections

        center = np.mean(points, axis=0)

        # Create Grain object directly
        new_grain = Grain_functions.Grain(
            points,
            contour_array,
            center,
            len(points),
            confidences[i],
            i + 1
        )
        Seg_Grain.append(new_grain)

    # 3. Fast Sorting
    Seg_Grain.sort(key=lambda x: x.size, reverse=True)

    # 4. Handle Overlaps (DO NOT use deepcopy here)
    # Instead of deepcopy, just pass the list.
    overlapping_grains, ID_grains = Grain_functions.find_overlapping_grains(Seg_Grain)
    Grain_functions.handle_overlapping_grains(overlapping_grains, SizeIm)

    # Generate grain matrix and mask
    grain_matrix, mask = Grain_functions.generate_grain_matrix(Seg_Grain, SizeIm)
    grain_matrix2 = Grain_functions.fill_contour_gaps(Seg_Grain, grain_matrix, mask, SizeIm)
    grain_matrix2 = Grain_functions.add_missing_regions(grain_matrix2, Seg_Grain, SizeIm)
    contour_mask = Grain_functions.create_contour_mask(grain_matrix2)
    contour_points_grains = find_contour_final(contour_mask, 2)
    contour_mask = contour_mask * 255
    contour_mask = np.rot90(contour_mask)
    contour_mask = np.flipud(contour_mask)

    return contour_points_grains, confidences, contour_mask

def process_twins(twins_dir, image_path, SizeIm):
    """
    Process twin contours and generate twin mask.

    Args:
        twins_dir (str): Directory containing YOLO prediction results for twins.
        image_path (str): Path to the input image.
        SizeIm (list): Size of the image as [width, height].

    Returns:
        tuple: (twins, confi, twin_image) where `twins` is a list of twin contours, `confi` is their confidence scores, and `twin_image` is the twin mask.
    """
    twins_labels_predict = os.path.join(twins_dir, 'labels')

    if image_path:
        twin_label_name = get_file_name_without_extension(image_path) + '.txt'
    else:
        twin_label_name = "image0.txt"
    twins_file = os.path.join(twins_labels_predict, twin_label_name)

    if not os.path.exists(twins_file):
        raise FileNotFoundError(f"Twins label file not found: {twins_file}")

    twins, confi = Grain_functions.read_contours(twins_file, SizeIm, 0)

    mask = np.zeros((SizeIm[1], SizeIm[0]), dtype=np.uint8)
    twins_cop = [np.array(twin, dtype=np.int32) for twin in twins]
    cv2.drawContours(mask, twins_cop, -1, (255), thickness=cv2.FILLED)

    twin_image = ~mask
    twin_image = twin_image.astype(np.uint8)

    if twin_image.max() == 1:
        twin_image = twin_image * 255

    rgb_image = cv2.cvtColor(twin_image, cv2.COLOR_GRAY2BGR)
    rgb_image[twin_image < 255] = [0, 0, 255]

    return twins, confi, twin_image

def save_results(final_folder, image_name, contour_mask, twin_image, SizeIm, segmented_contours, twins):
    """
    Save results to files.

    Args:
        final_folder (str): Path to the final folder.
        image_name (str): Name of the image.
        contour_mask (numpy.ndarray): Contour mask for grains.
        twin_image (numpy.ndarray): Twin mask.
        SizeIm (list): Size of the image as [width, height].
        segmented_contours (list): List of segmented grain contours.
        twins (list): List of twin contours.
    """
    label_folder = os.path.join(final_folder, "labels")
    label_name = get_file_name_without_extension(image_name) + '.txt'
    label_file = os.path.join(label_folder, label_name)

    # Save grain contours
    with open(label_file, 'w') as file:
        for contour_points in segmented_contours:
            file.write('1 ')
            if isinstance(contour_points, tuple) and len(contour_points) == 4:
                file.write(f'{contour_points[0] / SizeIm[0]} {contour_points[1] / SizeIm[1]} '
                           f'{contour_points[2] / SizeIm[0]} {contour_points[3] / (SizeIm[1] - 1)}\n')
            elif isinstance(contour_points, list):
                file.write(' '.join(
                    f'{point[0] / (SizeIm[1] - 1)} {point[1] / (SizeIm[0] - 1)}' for point in contour_points))
                file.write('\n')
            else:
                raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")

    # Save twin contours
    with open(os.path.join(label_folder, "twins_" + label_name), 'w') as file:
        for contour_points in twins:
            contours_twins = contour_points.tolist()
            file.write('2 ')
            if isinstance(contours_twins, tuple) and len(contours_twins) == 4:
                file.write(
                    f'{contours_twins[0] / (SizeIm[0] - 1)} {contours_twins[1] / (SizeIm[0] - 1)} '
                    f'{contours_twins[2] / (SizeIm[0] - 1)} {contours_twins[3] / (SizeIm[0] - 1)}\n')
            elif isinstance(contours_twins, list):
                file.write(' '.join(
                    f'{point[0] / (SizeIm[0] - 1)} {point[1] / (SizeIm[1] - 1)}' for point in contours_twins))
                file.write('\n')
            else:
                raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")

    # Save masks
    mask_folder = os.path.join(final_folder, "masks")
    cv2.imwrite(os.path.join(mask_folder, 'Grains', image_name), ~contour_mask)
    cv2.imwrite(os.path.join(mask_folder, 'Twins', image_name), twin_image)

    # Create and save the "Both" mask
    original = ~contour_mask

    final_image = np.ones_like(cv2.imread(os.path.join(mask_folder, 'Grains', image_name)), dtype=np.uint8) * 255
    twin_mask = twin_image < 255
    black_mask = np.all(original == [0, 0, 0], axis=2) if len(original.shape) == 3 else (original == 0)

    final_image[twin_mask] = [0, 0, 255]
    final_image[black_mask] = [0, 0, 0]
    cv2.imwrite(os.path.join(mask_folder, 'Both', image_name), final_image)

def simplify_method(image_path):
    """
    Main method to simplify the image by processing grains and twins.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Final processed image.
    """
    current_path = os.getcwd()
    models_path = os.path.join(current_path, "models")
    grains_model = os.path.join(models_path, "Grains_Model.pt")
    twins_model = os.path.join(models_path, "Twins_Model.pt")
    results_folder = os.path.join(current_path, "segmentation_results")

    file_name = get_file_name_without_extension(image_path)
    final_folder = os.path.join(results_folder, file_name)

    # Create directories
    create_directories(final_folder)

    # Predict grains and twins in parallel
    predict_base_dir = os.path.join(current_path, 'runs', 'segment')

    grains_dir, twins_dir = predict_grains_and_twins(image_path, grains_model, twins_model, predict_base_dir)

    # Get image dimensions
    image = cv2.imread(image_path)
    [m1n, n1n, d] = np.shape(image)
    SizeIm = [n1n, m1n]

    # Process grains and twins
    contour_points_grains, confidences, contour_mask = process_grains(grains_dir, image_path, SizeIm)
    twins, confi, twin_image = process_twins(twins_dir, image_path, SizeIm)

    # Save results
    image_name = get_file_name_without_extension(image_path) + '.png'
    save_results(final_folder, image_name, contour_mask, twin_image, SizeIm, contour_points_grains, twins)

    # Apply filters and save
    grains_image_path = os.path.join(final_folder, "masks", "Grains", image_name)
    grains_filtered = simple_filtered(grains_image_path)
    cv2.imwrite(grains_image_path, grains_filtered)

    both_image_path = os.path.join(final_folder, "masks", "Both", image_name)
    both_filtered = simple_filtered(both_image_path)
    both_advanced_filtered = advanced_filtered(both_filtered)
    cv2.imwrite(both_image_path, both_advanced_filtered)

    # Return final image
    final_image = cv2.imread(os.path.join(final_folder, "masks", "Both", image_name))
    return final_image

def amplify_method(image_path, original_path):


     # INITIALIZATION
    print("INITIALIZATION")
    start_time = time.time()
    current_path = os.getcwd()
    try:
        orientation_path = select_orientation_folder(original_path, current_path)
        pseudo_images, SizeIm = pseudo_imgs_generator(orientation_path, 0)
        pseudo_images.append(cv2.imread(image_path))


    except Exception as e:
        print(f"An error occurred: {e}")


    models_path = os.path.join(current_path, "models")
    grains_model = os.path.join(models_path, "Grains_Model.pt")
    twins_model = os.path.join(models_path, "Twins_Model.pt")
    m_twins = YOLO(twins_model)
    m_grains = YOLO(grains_model)
    results_folder = os.path.join(current_path, "segmentation_results")

    file_name = get_file_name_without_extension(image_path)
    final_folder = os.path.join(results_folder, file_name)

    # Create directories
    create_directories(final_folder)

    # Predict grains and twins in parallel
    predict_base_dir = os.path.join(current_path, 'runs', 'segment')

    os.makedirs(predict_base_dir, exist_ok=True)
    end_time = time.time()
    print(f"END INITIALIZATION: {end_time-start_time:.4f} seconds")

    print("ML PREDICTION")
    start_time = time.time()

    # Get image dimensions
    image = cv2.imread(image_path)
    m1n, n1n, d = image.shape
    SizeIm = [n1n, m1n]

    with ThreadPoolExecutor() as executor:
        # Submit both prediction tasks to the executor
        grains_future = executor.submit(predict_and_process_grains, pseudo_images[3], m_grains, SizeIm)
        twins_future = executor.submit(predict_twins_fast, pseudo_images, m_twins)

        # Retrieve the results
        contour_points_grains, confidences, grain_matrix = grains_future.result()
        final_twins, twins = twins_future.result()

    end_time = time.time()
    print(f"END ML PREDICTION: {end_time - start_time:.4f} seconds")

    # Save results
    print("SAVING PREDICTION")
    start_time = time.time()
    image_name = get_file_name_without_extension(image_path) + '.png'
    save_results(final_folder, image_name, grain_matrix, final_twins, SizeIm, contour_points_grains, twins)

    # Apply filters and save
    grains_image_path = os.path.join(final_folder, "masks", "Grains", image_name)
    twins_image_path = os.path.join(final_folder, "masks", "Twins", image_name)
    grains_filtered = simple_filtered(grains_image_path)
    cv2.imwrite(grains_image_path, grains_filtered)

    both_image_path = os.path.join(final_folder, "masks", "Both", image_name)
    both_filtered = simple_filtered(both_image_path)
    both_advanced_filtered = advanced_filtered(both_filtered)
    cv2.imwrite(both_image_path, both_advanced_filtered)
    end_time = time.time()
    print(f"END SAVING PREDICTION: {end_time - start_time:.4f} seconds")
    print("BIGGEST CHUNK STARTING")
    start_time = time.time()
    final_image, plm_map = orientation_and_classification(orientation_path, grains_image_path, twins_image_path, image_name)
    # Return final image
    end_time = time.time()
    print(f"END BIGGEST CHUNK: {end_time - start_time:.4f} seconds")
    return final_image, plm_map
