import cv2
import numpy as np
import tkinter as tk
import os
import random
from tkinter import filedialog, messagebox
import glob
import re
import math


from skimage.morphology import skeletonize, thin
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize

import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from skimage.draw import disk as draw_disk
from skimage.morphology import erosion, square
from collections import Counter

# Numerical and Image Processing
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull

# Skimage
from skimage import io, img_as_ubyte
from skimage.filters.rank import modal
from skimage.morphology import square, skeletonize, dilation, disk
from skimage.measure import label, regionprops, find_contours

# Plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# Geometry
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import make_valid
from shapely.errors import ShapelyError

# External Tools
from ultralytics import YOLO
import alphashape
from tqdm import tqdm

# Project-Specific
from source_code.Grain_functions import find_grain_by_ID, find_grain_by_ID_index, merge_grain, decompose_twins_grains, decompose_twins_grains_2
from source_code.Dataset_Generator import get_latest_predict_dir, prepare_data, read_contours, find_contour_final
from source_code import Grain_functions
# from Test import get_integer_points_inside_contour  # Optional
from source_code.pseudoimage import apply_clahe, adjust_contrast, normalize_images_all
from matplotlib.patches import Ellipse

# %% Functions

def check_peaks(grains):
    """
    Verifies all Inclination values are valid: not None, finite, and numeric (float or convertible to float).
    Inclination can be zero.
    """
    for i, grain in enumerate(grains):
        incl = grain.Inclination

        if incl is None:
            print(f"❌ Error: Grain ID {i} has Inclination set to None.")
            raise ValueError("Inclination is None.")

        try:
            value = float(incl)
        except (TypeError, ValueError):
            print(f"❌ Error: Grain ID {i} has non-numeric Inclination value: {incl}")
            raise ValueError("Inclination is not a number.")

        if not np.isfinite(value):
            print(f"❌ Error: Grain ID {i} has non-finite Inclination value: {value}")
            raise ValueError("Inclination is not finite.")

    print("✅ All grains passed Inclination integrity check.")

def gray_mean(grain_stats, folder):

    files = sorted(glob.glob(os.path.join(folder, '*.png')), key=extract_number2)
    num_grains = len(grain_stats)
    num_images = len(files)

    gray_mean_array = np.zeros((num_grains, num_images))
    grain_positions = np.zeros((num_grains, 2))
    count = np.zeros(num_grains)
    images = []

    # -------------------- IMAGE PREPROCESSING --------------------
    for m in range(num_images):
        image = cv2.imread(files[m], cv2.IMREAD_GRAYSCALE)
        images.append(image)
    images_final = normalize_images_all(images)

    # -------------------- GRAIN PROCESSING --------------------
    for img_idx in range(num_images):
        image = images_final[img_idx]
        for i, grain in enumerate(grain_stats):
            pixels = np.array(grain.PixelList)
            y = pixels[:, 1].astype(int)
            x = pixels[:, 0].astype(int)

            gray_values = image[y, x]
            gray_mean_array[i, img_idx] = np.mean(gray_values)
            grain_positions[i] = [np.mean(x), np.mean(y)]

    # -------------------- UPDATE GRAIN STATS --------------------
    for i, grain in enumerate(grain_stats):
        grain.Position = grain_positions[i]
        grain.GrayMean = gray_mean_array[i]
        count[i] = np.sum(grain.GrayMean < 10)

    result = np.mean(gray_mean_array, axis=0)

    return grain_stats, result


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

def color_mask(image, color, tol=10):
    return np.all(np.abs(image - color) <= tol, axis=2)

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


def Peaks(Grainstats, initial_s=50, s_increment=20, max_threshold=254):
    """
    Estimate peak orientations for grains using a smoothing spline.
    If the maximum amplitude across all grains exceeds `max_threshold`,
    the smoothing parameter `s` is increased by `s_increment` and recalculation is performed.

    Parameters:
    - Grainstats: list of objects with attributes GrayMean (list of gray values) and ID,
                  and method set_orientation(azimuth, inclination)
    - initial_s: starting smoothing factor for UnivariateSpline
    - s_increment: how much to increase smoothing factor if threshold exceeded
    - max_threshold: amplitude threshold to adjust smoothing

    Returns:
    - Grainstats with orientations set
    """
    s_val = initial_s
    while True:
        Amps = []
        locations = []

        # Loop over grains to compute amplitudes and locations
        for grain in Grainstats:

            gray = np.array(grain.GrayMean)
            # handle 37-point wrap-around
            if gray.size == 37:
                gray = gray[:-1]
            if gray.size == 19:
                gray = gray[:-1]
            # extend data for circularity
            testGray_extended = np.tile(gray, 2)

            x2 = np.arange(0, 360, 10)
            x1_full = np.arange(0, 360, 0.01)

            spline = UnivariateSpline(x2, testGray_extended, s=s_val)
            smoothed_full = spline(x1_full)
            smoothed = smoothed_full[:18000]

            # find peaks
            height_thresh = smoothed.min() + 0.6 * (smoothed.max() - smoothed.min())
            peaks, info = find_peaks(smoothed, height=height_thresh)
            pks = info.get('peak_heights', [])
            locs = peaks.tolist()
            ok = 0
            while ok == 0:
                if pks is None or len(pks) == 0:
                    # try shifted
                    shifted = np.roll(smoothed, 90)
                    peaks, info = find_peaks(shifted, height=height_thresh)
                    pks = info.get('peak_heights', [])
                    if pks is not None or len(pks) > 0:
                        locs = (peaks - 90).tolist()
                        toto = locs[0] / 100
                        locs[0] = toto
                        ok = 1
                    else:
                        pks = [smoothed[0]]
                        locs = [0]
                        ok = 1
                else:
                    toto = locs[0] / 100
                    locs[0] = toto
                    ok = 1

            # store
            Amps.append(np.max(pks) - np.min(smoothed))
            locations.append(locs)

        maxAmp = 255
        # check threshold
        if maxAmp > max_threshold:
            print('TO HIGH')
            # increase smoothing and retry
            s_val += s_increment
        else:
            break

    # Final assignment
    for amp, locs, grain in zip(Amps, locations, Grainstats):
        inclination = 0 if amp <= 0 else (amp * 90 / maxAmp)
        azimuth = (locs[0] - 45) % 180
        grain.set_orientation(azimuth, inclination)

    return Grainstats

def check_grains(grains):
    """
    Verifies all PixelList arrays are of integer type.
    If any non-integer or invalid entries are found, the function raises an error and exits.
    """
    for grain in grains:
        pixels = np.array(grain.PixelList)

        # Check if all values are finite and integers
        if not np.isfinite(pixels).all() or not np.issubdtype(pixels.dtype, np.integer):
            print(f"❌ Error: Grain ID {grain.ID} has invalid or non-integer pixel values.")
            raise ValueError("PixelList contains non-integer or invalid entries.")

    print("✅ All grains passed pixel integrity check.")


def Neighbours(Grainstats, B):
    sizeA = B.shape
    Y, X = np.meshgrid(np.arange(sizeA[1]), np.arange(sizeA[0]))  # X: cols, Y: rows
    listePixel = []
    for i, grain in enumerate(Grainstats):
        pixel_list = np.array(grain.PixelList)
        grain_array = np.zeros((pixel_list.shape[0], 3), dtype=int)
        grain_array[:, 0:2] = pixel_list
        grain_array[:, 2] = grain.ID  # Grain ID (1-based like MATLAB)
        listePixel.append(grain_array)

    # Combine all positions
    AllPosition = np.vstack(listePixel)
    # Create Zfinal grain map
    Zfinal = np.zeros(np.flip(sizeA), dtype=int)
    for row in AllPosition:
        x, y, grain_id = row
        Zfinal[x, y] = grain_id

    # Neighbour extraction
    for i, grain in tqdm(enumerate(Grainstats), total=len(Grainstats), desc="Processing grains"):

        if grain.IsTwin == True or grain.HaveFriends == True:
            grain.Neighbours = []

            # Create binary mask of the grain
            Mask = np.zeros(np.flip(sizeA), dtype=np.uint8)
            pixel_list = np.array(grain.PixelList)
            Mask[pixel_list[:, 0], pixel_list[:, 1]] = 1

            GrainIDUnique = set()
            for m in range(2, 5, 2):  # m = 2, 4, 6, 8, 10
                selem = disk(m)
                Mask_dilated = dilation(Mask, selem)
                Mask_border = (Mask_dilated.astype(int) - Mask.astype(int)).astype(bool)
                BorderMask = Zfinal * Mask_border
                new_neighbours = np.unique(BorderMask)
                GrainIDUnique.update(new_neighbours[new_neighbours != 0])

            # Remove self if present
            GrainIDUnique.discard(i + 1)
            grain.set_neighbours(list(GrainIDUnique))

    return Grainstats


def Neighbours2(Grainstats, B):
    sizeA = B.shape
    Y, X = np.meshgrid(np.arange(sizeA[1]), np.arange(sizeA[0]))  # X: cols, Y: rows
    listePixel = []

    for i, grain in enumerate(Grainstats):
        pixel_list = np.array(grain.PixelList, dtype=int)  # Ensures integers
        grain_array = np.zeros((pixel_list.shape[0], 3), dtype=int)
        grain_array[:, 0:2] = pixel_list
        grain_array[:, 2] = grain.ID  # Grain ID (1-based like MATLAB)
        listePixel.append(grain_array)

    # Combine all positions
    AllPosition = np.vstack(listePixel)

    # Create Zfinal grain map
    Zfinal = np.zeros(np.flip(sizeA), dtype=int)
    for row in AllPosition:
        x, y, grain_id = row
        Zfinal[x, y] = grain_id

    # Neighbour extraction
    for i, grain in tqdm(enumerate(Grainstats), total=len(Grainstats), desc="Processing grains"):

        if grain.IsTwin or grain.HaveFriends:

            grain.Neighbours = []
            # Create binary mask of the grain
            Mask = np.zeros(np.flip(sizeA), dtype=np.uint8)
            pixel_list = np.array(grain.PixelList, dtype=int)  # Ensures integers
            Mask[pixel_list[:, 0], pixel_list[:, 1]] = 1

            GrainIDUnique = set()
            for m in range(2, 5, 2):  # m = 2, 4, 6, 8, 10
                selem = disk(m)
                Mask_dilated = dilation(Mask, selem)
                Mask_border = (Mask_dilated.astype(int) - Mask.astype(int)).astype(bool)
                BorderMask = Zfinal * Mask_border
                new_neighbours = np.unique(BorderMask)
                GrainIDUnique.update(new_neighbours[new_neighbours != 0])

            # Remove self if present
            GrainIDUnique.discard(i + 1)
            grain.set_neighbours(list(GrainIDUnique))

    return Grainstats


# %% MAIN FUNCTION
def orientation_and_classification(orientation_path, grains_image_path, twins_image_path, image_name):
    path_gr = grains_image_path
    path_tw = twins_image_path
    img_study = image_name
    grains = cv2.imread(grains_image_path, cv2.COLOR_BGR2GRAY)
    nothing, size = Grain_functions.image_size(grains_image_path)
    twins = cv2.imread(twins_image_path, cv2.COLOR_BGR2GRAY)
    skeleton_grains = np.uint8(skeletonize(np.uint8(~grains)))

    skeleton_twins = np.uint8(skeletonize(np.uint8(~twins)))

    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(skeleton_twins, kernel, iterations=1)

    bool_ske = ~dilated_image.astype(bool)
    bool_ske_tw = bool_ske
    labeled_array_ske, num_features_ske = label(~bool_ske, return_num=True)
    print(np.shape(bool_ske))
    print(np.shape(labeled_array_ske))
    contours = find_contours(labeled_array_ske, 0.5)
    contour_twins = np.zeros_like(grains, dtype=np.uint8)
    combined_image = np.zeros_like(grains, dtype=np.uint8)
    skeleton_grains[skeleton_grains == 1] = 255
    contour_points_list = []
    Twins = []
    for contour in contours:
        contour = np.round(contour).astype(int)
        contour = np.flip(contour)
        cv2.fillPoly(skeleton_grains, [contour], 0)  # Fill the interior of the contour with 0
        cv2.drawContours(skeleton_grains, [contour], -1, (255), thickness=1)  # Draw the contour lines
        contour_points = []
        contour = np.flip(contour)
        for point in contour:
            y, x = point

            corrected_x = max(0, round(x))
            corrected_y = max(0, round(y))
            contour_points.append((corrected_x, corrected_y))
            contour_twins[round(y), round(x)] = 255
        contour_points_list.append(contour_points)
    for i, contour in enumerate(contour_points_list):
        contour2 = np.array(contour, np.int32)
        points_inside = Grain_functions.get_integer_points_inside_contour(contour2)
        points = np.array(points_inside, dtype=np.int32)
        contour_array = np.array(contour, dtype=np.int32)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        gr = Grain_functions.Grain(points, contour_array, (center_x, center_y), len(points[:, 0]), 1, i + 1)
        gr.is_twinning(True)
        Twins.append(gr)

    combined_image[combined_image == 0] = skeleton_grains[combined_image == 0]

    skeleton_grains = np.uint8(skeletonize(np.uint8(combined_image)))
    skeleton = np.pad(skeleton_grains, pad_width=1, mode='constant', constant_values=1)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1)

    bool_ske = ~dilated_image.astype(bool)
    bool_ske_gr = bool_ske
    labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
    contours = []
    contours = find_contours(labeled_array_ske, 0.5)
    contour_twins = np.zeros_like(grains, dtype=np.uint8)
    # Create a combined image initialized with zeros
    combined_image = np.zeros_like(grains, dtype=np.uint8)
    skeleton_grains[skeleton_grains == 1] = 255
    contour_points_list = []
    for contour in contours:
        contour = np.round(contour).astype(int)
        contour_points = []

        for point in contour:
            y, x = point
            corrected_x = max(0, math.floor(x) - 1)
            corrected_y = max(0, math.floor(y) - 1)
            contour_points.append((corrected_x, corrected_y))

            # contour_image[math.floor(y)-1, math.floor(x)-1] = 255
        contour_points_list.append(contour_points)
    for i, contour in enumerate(contour_points_list):
        contour2 = np.array(contour, np.int32)
        points_inside = Grain_functions.get_integer_points_inside_contour(contour2)
        points = np.array(points_inside, dtype=np.int32)
        contour_array = np.array(contour, dtype=np.int32)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        gr = Grain_functions.Grain(points, contour_array, (center_x, center_y), len(points[:, 0]), 1, i + 1)
        Grains.append(gr)
    final_orientation = orientation_path
    check_grains(Grains)

    Grains2, result = gray_mean(Grains, final_orientation)

    Grains3 = Peaks(Grains2)
    # plot_grains_contour(Grains3, grains.shape)
    check_peaks(Grains3)
    grano = grains
    Grains4 = copy.deepcopy(Grains3)

    Grains4 = Neighbours(Grains4, grano)

    check_grains(Grains4)

    id_twins = []
    Error_Angle = 6
    A = []
    for gr in Grains4:
        if gr.IsTwin == True:
            id_twins.append(gr.ID)

            for m in range(len(gr.Neighbours)):
                studied_grain = Grains4[gr.Neighbours[m] - 1]
                phi1 = studied_grain.Azimuth
                Phi = studied_grain.Inclination
                id1 = studied_grain.ID

                for n in range(len(gr.Neighbours)):
                    if n != m:
                        A = []
                        phi1 = studied_grain.Azimuth
                        Phi = studied_grain.Inclination
                        id1 = studied_grain.ID
                        studied_grain2 = Grains4[gr.Neighbours[n] - 1]
                        phi11 = studied_grain2.Azimuth
                        Phi1 = studied_grain2.Inclination
                        id2 = studied_grain2.ID

                        phi21 = 0
                        if phi11 >= 90 and phi1 < 90:
                            A.append(final_angle(phi1, Phi, phi11, Phi1))
                            phi11 = 180 + phi11
                            A.append(final_angle(phi1, Phi, phi11, Phi1))
                            CosTheta = np.min(A)
                        elif phi1 >= 90 and phi11 < 90:
                            A.append(final_angle(phi1, Phi, phi11, Phi1))
                            phi1 = 180 + phi1
                            A.append(final_angle(phi1, Phi, phi11, Phi1))
                            CosTheta = np.min(A)
                        else:
                            CosTheta = final_angle(phi1, Phi, phi11, Phi1)
                        if CosTheta < 0:
                            CosTheta = 180 + CosTheta

                        CosTheta = CosTheta % 180
                        if CosTheta <= Error_Angle:
                            studied_grain.add_friends(studied_grain2.ID)

    Grains5 = copy.deepcopy(Grains4)
    Grains5 = Neighbours(Grains5, grano)

    check_grains(Grains5)
    Grains6 = copy.deepcopy(Grains5)

    check_grains(Grains6)
    Grains6, result = gray_mean(Grains6, final_orientation)

    Grains6 = Peaks(Grains6)

    check_grains(Grains6)
    check_peaks(Grains6)

    overlapping_grains, ID_grains = Grain_functions.find_overlapping_grains(Grains6)

    for i, grains_couple in tqdm(enumerate(overlapping_grains), total=len(overlapping_grains),
                                 desc="Processing grains"):
        if any(grain.AtRisk for grain in grains_couple):
            Grain_functions.remove_overlapping_pixels_ATRISK(grains_couple[0], grains_couple[1], size)

    new_grains = []
    for gr in Grains6:
        if gr.AtRisk and (len(gr.ContourPoints) == 0 or len(gr.PixelList) == 0):
            print(f"Removed grain ID {gr.ID} due to empty ContourPoints or PixelList.")
            continue
        new_grains.append(gr)

    Grains6 = new_grains
    for i, gr in enumerate(Grains6):
        if gr.AtRisk == True:
            print(i)
            # gr.PixelList = np.flip(gr.PixelList, axis=1)
            gr.ContourPoints = np.flip(gr.ContourPoints, axis=1)

    Grains7 = copy.deepcopy(Grains6)
    Grains7 = Neighbours2(Grains7, grano)

    check_grains(Grains7)

    grain_matrix = np.zeros(np.flip(grano.shape), dtype=np.int32)
    mask = np.zeros(np.flip(grano.shape), dtype=np.uint8)
    for grain in Grains7:

        for (x, y) in grain.PixelList:
            try:
                grain_matrix[x, y] = grain.ID
                mask[x, y] = 1
            except IndexError as e:
                print(f"IndexError: Grain ID {grain.ID} caused an out-of-bounds error at position ({x}, {y})")
                continue
    Grains7 = sorted(Grains7, key=lambda g: g.confidence, reverse=True)
    grain_matrix2 = copy.deepcopy(grain_matrix)

    for grain in Grains7:
        # Create an empty mask for the grain contour
        mask2 = np.zeros(np.flip(grano.shape), dtype=np.uint8)

        # Draw the contour of the grain on mask2
        cv2.drawContours(mask2, [grain.ContourPoints], -1, (1), thickness=1)

        # Dilate mask2 to cover potential gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask2 = cv2.dilate(mask2, kernel, iterations=1)

        # Find positions where mask2 - mask equals 1
        difference_mask = dilated_mask2 - mask

        # Update grain_matrix with the current grain ID for these positions
        grain_matrix2[difference_mask == 1] = grain.ID

    zero_mask = (grain_matrix2 == 0).astype(np.uint8)
    labeled_zeros, num_labels = label(zero_mask, return_num=True)

    new_grains = []
    max_existing_id = max(grain.ID for grain in Grains7)
    n = 1
    for region in regionprops(labeled_zeros):
        if region.area > 0:  # Ignore very small regions if needed
            # Extract the coordinates of the current region
            coords = region.coords
            new_id = max_existing_id + n + 1

            new_mask = np.zeros(np.flip(grano.shape), dtype=np.uint8)
            for (x, y) in coords:
                try:
                    new_mask[x, y] = 1
                except IndexError as e:
                    # print("toto")
                    continue  # Skip this pixel and continue with the next one
            # Update the grain_matrix with the new grain ID
            grain_matrix2[zero_mask == 1] = new_id
            n = n + 1

    kernel = np.ones((3, 3), np.uint8)
    smoothed_mask = cv2.morphologyEx(grain_matrix2.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)
    # Create a contour mask where the gradient is positive
    # Compute gradient of the smoothed mask
    gradient_x = cv2.Sobel(smoothed_mask, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed_mask, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Create a contour mask where the gradient is positive
    contour_mask = (gradient_magnitude > 0).astype(np.uint8)
    contour_mask_bool = contour_mask.astype(bool)

    skeletonized_contour_mask = skeletonize(contour_mask_bool).astype(np.uint8)
    # Dilate the contour mask to enhance contours
    kernel_dilate = np.ones((1, 1), np.uint8)
    dilated_contours = cv2.dilate(skeletonized_contour_mask, kernel_dilate, iterations=1)

    # Initialize an empty RGB image (black background)
    width, height = grano.shape
    # Create twin_contour_image with white background

    Grains8 = copy.deepcopy(Grains7)

    kernel = np.ones((3, 3), np.uint8)  # Or larger if you want more dilation
    # plot_grains(Grains7, grains.shape)
    for grain in Grains8:
        # 1. Create binary mask for this grai
        grain_mask = np.zeros(np.flip(grano.shape), dtype=np.uint8)
        for (x, y) in grain.PixelList:
            try:
                grain_mask[x, y] = 1
            except IndexError:
                continue  # Skip bad pixels

        # 2. Dilate the grain mask
        dilated_grain_mask = cv2.dilate(grain_mask, kernel, iterations=1)

        # 3. Find contours from the dilated mask
        contours, _ = cv2.findContours(
            dilated_grain_mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Optional: pick the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            arr2 = np.squeeze(largest_contour, axis=1)
            grain.DilatedContourPoints = arr2  # Add to grain object
            # 4. Create a filled mask from the dilated contour

        else:
            grain.DilatedContourPoints = grain.ContourPoints  # Fallback

        filled_mask = np.zeros_like(grain_mask, dtype=np.uint8)
        cv2.drawContours(filled_mask, [largest_contour], contourIdx=-1, color=1, thickness=-1)

        # 5. Extract pixels inside or on contour
        new_pixels = np.column_stack(np.where(filled_mask > 0))
        grain.PixelList = [(int(x), int(y)) for x, y in new_pixels]

    # Assuming image_shape is the same as your grain mask shape

    min_size = 10
    Grains8 = [gr for gr in Grains8 if len(gr.PixelList) > min_size]

    Grains9 = Neighbours2(Grains8, grano)

    Grains10 = copy.deepcopy(Grains9)

    Grains10 = delete_small_twins(Grains10)

    for gr in Grains10:
        gr.update_ellipsoid()

    # Extract sizes and lengths
    sizes = np.array([gr.size for gr in Grains10])
    lengths = np.array([2 * gr.l1 for gr in Grains10])

    # Compute averages
    Average_Size = sizes.mean() if len(sizes) > 0 else 0
    Average_length = lengths.mean() if len(lengths) > 0 else 0

    # Compute standard deviations
    Std_Size = sizes.std(ddof=1) if len(sizes) > 1 else 0  # ddof=1 for sample std
    Std_length = lengths.std(ddof=1) if len(lengths) > 1 else 0
    # plot_grains_ID(Grains10, grano.shape, Average_length, final_image, save_path = "grain_map.png")

    Grains10 = decompose_twins_grains(Grains10, grano, Average_length, Std_length)

    Grains10 = Neighbours3(Grains10, grano)

    Grains11 = copy.deepcopy(Grains10)

    Grains11, result = gray_mean_twin(Grains11, final_orientation)
    Grains11 = Peaks(Grains11)
    Grains11 = [gr for gr in Grains11 if gr.size > min_size]
    Zfinal = np.zeros(np.flip(grano.shape), dtype=int)

    for gr in Grains11:
        pixel_list = np.array(gr.PixelList)
        Zfinal[pixel_list[:, 0], pixel_list[:, 1]] = gr.ID

    Grains11 = find_parents_separate_twins(Grains11, np.flip(grano.shape), Zfinal, Average_Size, background=grano)
    Grains11 = decompose_twins_grains_2(Grains11, grano, Average_length, Std_length)

    Grains11, result = gray_mean_twin(Grains11, final_orientation)
    Grains11 = Peaks(Grains11)

    final_image = np.ones((height,width,3), dtype=np.uint8)*255
    twin_contour_image_black = np.ones((height, width, 3), dtype=np.uint8) * 255
    twin_contour_image_red = np.ones((height, width, 3), dtype=np.uint8) * 255
    twin_contour_image_blue = np.ones((height, width, 3), dtype=np.uint8) * 255
    twin_contour_image_green = np.ones((height, width, 3), dtype=np.uint8) * 255
    twin_contour_image_orange = np.ones((height, width, 3), dtype=np.uint8) * 255

    for grain in Grains11:
        grain.ContourLength = measure_contour_length(grain.DilatedContourPoints)


        try:
            cv2.drawContours(
                twin_contour_image_black,
                [grain.DilatedContourPoints],  # Must be a list of arrays
                contourIdx=-1,
                color=(0, 0, 0),  # Red in BGR (OpenCV)
                thickness=1
            )
        except Exception as e:
            print(f"Failed drawing contour for Grain ID {grain.ID}: {e}")

        if grain.IsTwin:

            if grain.TwinType == "Tension":
                try:
                    cv2.drawContours(
                        twin_contour_image_green,
                        [grain.DilatedContourPoints],  # Must be a list of arrays
                        contourIdx=-1,
                        color=(50, 205, 50),  # Red in BGR (OpenCV)
                        thickness=2
                    )
                except Exception as e:
                    print(f"Failed drawing contour for Grain ID {grain.ID}: {e}")
            if grain.TwinType == "Compression":
                try:
                    cv2.drawContours(
                        twin_contour_image_blue,
                        [grain.DilatedContourPoints],  # Must be a list of arrays
                        contourIdx=-1,
                        color=(0, 0, 255),  # Red in BGR (OpenCV)
                        thickness=2
                    )
                except Exception as e:
                    print(f"Failed drawing contour for Grain ID {grain.ID}: {e}")

            if grain.TwinType == "None":
                try:
                    cv2.drawContours(
                        twin_contour_image_red,
                        [grain.DilatedContourPoints],  # Must be a list of arrays
                        contourIdx=-1,
                        color=(255, 0, 0),  # Red in BGR (OpenCV)
                        thickness=2
                    )
                except Exception as e:
                    print(f"Failed drawing contour for Grain ID {grain.ID}: {e}")

    is_red = color_mask(twin_contour_image_red, np.array([255, 0, 0]))
    is_blue = color_mask(twin_contour_image_blue, np.array([0, 0, 255]))
    is_green = color_mask(twin_contour_image_green, np.array([50, 205, 50]))
    is_black = color_mask(twin_contour_image_black, np.array([0, 0, 0]))

    # Remove black pixels where red pixels exist
    is_black = np.logical_and(is_black, ~is_red)
    is_black = np.logical_and(is_black, ~is_blue)
    is_black = np.logical_and(is_black, ~is_green)

    # Now apply on final image
    final_image[is_black] = [0, 0, 0]
    final_image[is_red] = [255, 0, 0]
    final_image[is_green] = [50, 205, 50]
    final_image[is_blue] = [0, 0, 255]

    return final_image