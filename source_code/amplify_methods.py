import cv2
import numpy as np
import tkinter as tk
import os
import random
from tkinter import filedialog, messagebox
import glob
import re
import math
import copy
import time
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Geometry
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import make_valid
from shapely.errors import ShapelyError
from shapely.strtree import STRtree

# Skimage
from skimage import io, img_as_ubyte
from skimage.filters.rank import modal
from skimage.morphology import square, skeletonize, dilation, disk
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import skeletonize, thin
from skimage.draw import disk as draw_disk
from skimage.morphology import erosion, square
from skimage.draw import polygon

from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize

from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull

from ultralytics import YOLO
import alphashape
from tqdm import tqdm

# %% Own scripts
from source_code.Grain_functions import find_grain_by_ID, find_grain_by_ID_index, decompose_twins_grains, decompose_twins_grains_2
from source_code import Grain_functions
from source_code import Grain_Orientation
from source_code.pseudoimage import apply_clahe, adjust_contrast, normalize_images_all

# %% Functions

def safe_summarize_skeleton2(skeleton_img):
    """
    Safely summarize a skeletonized image.
    Keeps only the longest branch if multiple are present.

    Args:
        skeleton_img (np.ndarray): Binary skeleton image.

    Returns:
        pd.DataFrame | None: Summary of longest branch, or None if empty/error.
    """
    # Check if skeleton has any nonzero pixels
    if not np.any(skeleton_img):
        print("⚠ Skeleton is empty, skipping.")
        return None

    try:
        # Summarize skeleton
        branch_data = summarize(Skeleton(skeleton_img), separator='_')

        if branch_data is None or len(branch_data) == 0:
            return None

        # If multiple branches, keep only the longest
        if len(branch_data) > 1:
            longest_idx = branch_data['branch_distance'].idxmax()
            branch_data = branch_data.loc[[longest_idx]]

        return branch_data.reset_index(drop=True)

    except ValueError as e:
        print(f"⚠ Error processing skeleton: {e}")
        return None

def grain_twin_analysis(grain, grains, image_shape, Zfinal, Average_Size, background=None):
    skeleton = get_skeleton(grain, image_shape)
    skel, branches = get_branches(skeleton)

    branch_data = safe_summarize_skeleton2(skeleton)

    type_issue = -1

    endpoints, centroido = get_branch_endpoints_centroid(skel, branch_data, grain)
    neighbour_ids = grain.Neighbours
    centroids = compute_centroids(neighbour_ids, grains)
    centroids2 = copy.deepcopy(centroids)
    branch_results = []

    neighs_left_ID = []
    neighs_right_ID = []

    Parents = False
    if grain.ID == 94:
        piopipo = 3
    neighs_left = []
    neighs_right = []
    neighs_left_length = []
    neighs_right_length = []
    points = endpoints[0]

    for nid, centroid in centroids.items():
        if nid == 70:
            kk = 1
        if is_projection_inside_segment(np.array(points[0]), np.array(points[1]), centroid):
            length, left = is_left_or_right(np.array(points[0]), np.array(points[1]), centroid)
            if left:
                neighs_left_length.append(length)
                neighs_left_ID.append(nid)
                neighs_left.append(find_grain_by_ID(grains, nid))

            else:
                neighs_right_length.append(length)
                neighs_right_ID.append(nid)
                neighs_right.append(find_grain_by_ID(grains, nid))

        if nid in centroids2:
            del centroids2[nid]

    if (len(neighs_left) == 1 and len(neighs_right) == 1):
        Azimuth_P, Incli_P, Parents = check_friends(neighs_left[0], neighs_right[0])

        if Parents == True:
            grain, miso, type_error = check_twin_type(grain, Azimuth_P, Incli_P)
            grain.MisOrientation = miso
            type_issue = type_error
        else:
            type_issue = 2

    elif (len(neighs_left) == 0 and len(neighs_right) == 1) or (len(neighs_left) == 1 and len(neighs_right) == 0):
        if len(neighs_right) == 1:
            Azimuth_P = neighs_right[0].Azimuth
            Incli_P = neighs_right[0].Inclination

            Parents = True
        if len(neighs_left) == 1:
            Azimuth_P = neighs_left[0].Azimuth
            Incli_P = neighs_left[0].Inclination

            Parents = True
        if Parents == True:
            grain, miso, type_error = check_twin_type(grain, Azimuth_P, Incli_P)
            grain.MisOrientation = miso
            type_issue = type_error

    elif (len(neighs_left) >= 2 or len(neighs_right) >= 2):
        type_issue = 1

    elif (len(neighs_left) == 0 and len(neighs_right) == 0):
        type_issue = 0

    return grain, type_issue

def separate_twin(grain, neighs_left, neighs_right, image_shape, maxID, skeleton_grains):
    mask = np.zeros(image_shape)

    pixel_list = np.array(neighs_left.PixelList, dtype=int)  # Ensures integers
    pixel_list2 = np.array(neighs_right.PixelList, dtype=int)  # Ensures integers

    pixel_list = np.concatenate((pixel_list, pixel_list2))

    hull = ConvexHull(pixel_list)
    # Extract hull vertices
    hull_points = pixel_list[hull.vertices]

    # Separate coordinates (Note: PixelList may be in (row, col) = (y, x) order)
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], shape=image_shape)

    # Fill mask inside hull
    mask[rr, cc] = 1

    mask2 = np.zeros(image_shape)
    pixel_list = np.array(grain.PixelList, dtype=int)  # Ensures integers
    mask2[pixel_list[:, 0], pixel_list[:, 1]] = 1

    mask_total = mask + mask2

    # Get the coordinates where mask_total equals 2
    rows, cols = np.where(mask_total == 2)

    mask3 = np.zeros(image_shape)
    mask3[rows, cols] = 1
    contours = find_contours(mask3, 0.5)
    if len(contours) > 1:
        # Calculate the length of each contour (using number of points as a simple proxy)
        contour_lengths = [len(contour) for contour in contours]
        # Find the index of the longest contour
        longest_contour_index = np.argmax(contour_lengths)
        # Keep only the longest contour
        contours = [contours[longest_contour_index]]

    new_granulo = []
    contour_points_list = []
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

        contour_points_list.append(contour_points)

    for i, contour in enumerate(contour_points_list):
        contour2 = np.array(contour, np.int32)
        points_inside = Grain_functions.get_integer_points_inside_contour(contour2)
        points = np.array(points_inside, dtype=np.int32)
        final_points = (points[:, 1], points[:, 0])
        final_pts = np.array(final_points)
        contour_array = np.array(contour, dtype=np.int32)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        gr = Grain_functions.Grain(final_pts, contour_array, (center_y, center_x), len(points[:, 0]), 1, grain.ID)
        gr.PixelList = [(int(y), int(x)) for x, y in points]
        gr.is_twinning(True)
        gr.DilatedContourPoints = contour_array
        gr.Dad = neighs_left.ID
        gr.Mum = neighs_right.ID
        gr.Neighbours = grain.Neighbours

    the_goat = gr
    mask2[mask_total == 2] = 0
    contours = find_contours(mask2, 0.5)
    contour_points_list = []
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

        contour_points_list.append(contour_points)
    for i, contour in enumerate(contour_points_list):
        contour2 = np.array(contour, np.int32)
        points_inside = Grain_functions.get_integer_points_inside_contour(contour2)
        points = np.array(points_inside, dtype=np.int32)

        contour_array = np.array(contour, dtype=np.int32)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        if len(points[:, 0]) >= 10:
            gr = Grain_functions.Grain(final_pts, contour_array, (center_y, center_x), len(points[:, 0]), 1,
                                       maxID + i + 1)
            gr.PixelList = [(int(y), int(x)) for x, y in points]
            gr.DilatedContourPoints = contour_array
            gr.is_twinning(True)
            gr.Neighbours = grain.Neighbours
            new_granulo.append(gr)

    # Stack the coordinates to get a list of (row, col) pairs

    return the_goat, new_granulo


def find_max_ID(grains):
    max_ID = 0
    for gr in grains:
        if gr.ID > max_ID:
            max_ID = gr.ID

    return max_ID


def misorientation_angle(euler1, euler2, m, degrees=True):
    """
    Compute misorientation angle between two orientations given as Euler triplets.
    """

    euler1 = np.array(euler1)
    euler2 = np.array(euler2)
    if m == 1:
        euler1[0] = euler1[0] + 180
    if m == 2:
        euler1[0] = euler1[0] + 180
        euler2[0] = euler2[0] + 180
    if m == 3:
        # euler1[0] = euler1[0] + 180
        euler2[0] = euler2[0] + 180
    if m == 4:
        euler2[1] = 180 - euler2[1]
    if m == 5:
        euler1[0] = euler1[0] + 180
        euler2[1] = 180 - euler2[1]
    if m == 6:
        euler1[0] = euler1[0] + 180
        euler2[0] = euler2[0] + 180
        euler2[1] = 180 - euler2[1]
    if m == 7:
        euler2[0] = euler2[0] + 180
        euler2[1] = 180 - euler2[1]
    if m == 8:
        euler1[1] = 180 - euler1[1]
    if m == 9:
        euler1[0] = euler1[0] + 180
        euler1[1] = 180 - euler1[1]
    if m == 10:
        euler1[0] = euler1[0] + 180
        euler2[0] = euler2[0] + 180
        euler1[1] = 180 - euler1[1]
    if m == 11:
        euler2[0] = euler2[0] + 180
        euler1[1] = 180 - euler1[1]
    if m == 12:
        euler1[1] = 180 - euler1[1]
        euler2[1] = 180 - euler2[1]
    if m == 13:
        euler1[0] = euler1[0] + 180
        euler1[1] = 180 - euler1[1]
        euler2[1] = 180 - euler2[1]
    if m == 14:
        euler1[0] = euler1[0] + 180
        euler2[0] = euler2[0] + 180
        euler1[1] = 180 - euler1[1]
        euler2[1] = 180 - euler2[1]
    if m == 15:
        euler2[0] = euler2[0] + 180
        euler1[1] = 180 - euler1[1]
        euler2[1] = 180 - euler2[1]

    euler1 = tuple(euler1)
    euler2 = tuple(euler2)

    error_angle = final_angle(euler1[0], euler1[1], euler2[0], euler2[1])

    return error_angle

def rotx(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotz(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


# --- Step 5: Project and classify neighbour ---
def is_projection_inside_segment(pt1, pt2, centroid, tol=1e-2):
    vec = pt2 - pt1
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    direction = pt2 - pt1
    shortened_pt1 = pt2 - 0.95 * direction
    shortened_pt2 = pt1 + 0.95 * direction

    vec2 = shortened_pt2 - shortened_pt1
    norm2 = np.linalg.norm(vec2)
    vec_norm = vec2 / norm2
    proj_length = np.dot(centroid - shortened_pt1, vec_norm)
    proj_point = shortened_pt1 + proj_length * vec_norm
    return 0 - tol <= proj_length <= norm2


def is_left_or_right(pt1, pt2, centroid, tol=1e-2):
    vec = pt2 - pt1
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    direction = pt2 - pt1
    shortened_pt1 = pt2 - 0.95 * direction
    shortened_pt2 = pt1 + 0.95 * direction

    vec2 = shortened_pt2 - shortened_pt1
    norm2 = np.linalg.norm(vec2)
    vec_norm2 = vec2 / norm2

    vec_norm = vec / np.linalg.norm(vec)

    proj_length = np.dot(centroid - shortened_pt1, vec_norm2)
    proj_point = shortened_pt1 + proj_length * vec_norm2

    vec3 = centroid - proj_point
    norm3 = np.linalg.norm(vec3)
    final_vec3 = vec3 / norm3

    k = np.cross(np.array([vec_norm2[0], vec_norm2[1], 0]), np.array([final_vec3[0], final_vec3[1], 0]))

    if k[2] >= 0:
        return proj_length, True
    else:
        return proj_length, False


def misorentation_between_angles(grain1, grain2):
    A = []

    phi1 = grain1.Azimuth
    Phi = grain1.Inclination

    phi11 = grain2.Azimuth
    Phi1 = grain2.Inclination

    Euler1 = (phi11, Phi1, 0)
    Euler2 = (phi1, Phi1, 0)

    angles = np.zeros(16)

    for poss in range(16):
        angles[poss] = misorientation_angle(Euler1, Euler2, poss, degrees=True)

    return angles


def check_twin_type(grain, Azimuth_P, Incli_P, Error_Angle=10):
    A = []

    phi1 = grain.Azimuth
    Phi = grain.Inclination

    phi11 = Azimuth_P
    Phi1 = Incli_P

    Euler1 = (Azimuth_P, Incli_P, 0)
    Euler2 = (phi1, Phi, 0)

    angle_C1 = 64.60
    angle_C2 = 57.05
    angle_T1 = 84.78
    angle_T2 = 35.10
    type_error = -1
    miso = []

    for poss in range(16):
        angle = misorientation_angle(Euler1, Euler2, poss, degrees=True)
        miso.append(angle)

    # Track what was found
    found_types = set()

    miso = np.array(miso)  # ensure it's an ndarray

    if np.any((miso <= angle_T1 + Error_Angle / 2) & (miso >= angle_T1 - Error_Angle / 2)):
        found_types.add("Tension")
    if np.any((miso <= angle_T2 + Error_Angle / 2) & (miso >= angle_T2 - Error_Angle / 2)):
        found_types.add("Tension")

    if np.any((miso <= angle_C1 + Error_Angle / 2) & (miso >= angle_C1 - Error_Angle / 2)):
        found_types.add("Compression")
    if np.any((miso <= angle_C2 + Error_Angle / 2) & (miso >= angle_C2 - Error_Angle / 2)):
        found_types.add("Compression")

    # Check results
    if len(found_types) == 0:
        type_error = 3

    if len(found_types) == 1:
        grain.TwinType = found_types.pop()

    if len(found_types) >= 2:
        type_error = 4

    return grain, miso, type_error


def check_friends(studied_grain, studied_grain2, Error_Angle=10):
    A = []

    phi1 = studied_grain.Azimuth
    Phi = studied_grain.Inclination
    id1 = studied_grain.ID
    phi11 = studied_grain2.Azimuth
    Phi1 = studied_grain2.Inclination
    id2 = studied_grain2.ID
    phi21 = 0
    Azimuth = 0
    Inclination = 0

    Euler1 = (phi1, Phi, 0)
    Euler2 = (phi11, Phi1, 0)

    TotalSize = studied_grain.size + studied_grain2.size

    for poss in range(16):
        A.append(misorientation_angle(Euler1, Euler2, poss, degrees=True))

    CosTheta = np.min(A)
    CosTheta = CosTheta % 180

    if CosTheta <= Error_Angle:
        # Choose the larger grain based on size
        if studied_grain.size >= studied_grain2.size:
            Azimuth = phi1
            Inclination = Phi
        else:
            Azimuth = phi11
            Inclination = Phi1

        return Azimuth, Inclination, True
    else:
        # Return the same larger-grain orientation even if condition fails
        if studied_grain.size >= studied_grain2.size:
            Azimuth = phi1
            Inclination = Phi
        else:
            Azimuth = phi11
            Inclination = Phi1

        return Azimuth, Inclination, False

def final_angle(azi_ejm, incli_ejm, azi_ebsd, incli_ebsd):
    point1 = np.array([0, 0, 1])  # Optical axis

    # EJM orientation
    PR = rotz(0) @ point1
    PR = rotx(incli_ejm) @ PR
    PR = rotz(azi_ejm) @ PR

    # EBSD orientation
    PRA = rotz(0) @ point1
    PRA = rotx(incli_ebsd) @ PRA
    PRA = rotz(azi_ebsd) @ PRA

    # Angle between vectors
    dot_product = np.clip(np.dot(PRA, PR), -1.0, 1.0)  # Clip to avoid numerical issues
    error_angle = np.rad2deg(np.arccos(dot_product))

    return error_angle

def final_angle_rot(azi_ejm, incli_ejm, rot_ejm, azi_ebsd, incli_ebsd, rot_ebsd):
    point1 = np.array([0, 0, 1])  # Optical axis

    # EJM orientation
    PR = rotz(rot_ejm) @ point1
    PR = rotx(incli_ejm) @ PR
    PR = rotz(azi_ejm) @ PR

    # EBSD orientation
    PRA = rotz(rot_ebsd) @ point1
    PRA = rotx(incli_ebsd) @ PRA
    PRA = rotz(azi_ebsd) @ PRA

    # Angle between vectors
    dot_product = np.clip(np.dot(PRA, PR), -1.0, 1.0)  # Clip to avoid numerical issues
    error_angle = np.rad2deg(np.arccos(dot_product))

    return error_angle

def subtract_contours_from_skeleton(skeleton, contours):
    """
    Remove filled contours from skeleton.
    """
    skeleton = skeleton.copy()

    for contour in contours:
        cv2.fillPoly(skeleton, [contour], 0)
        cv2.drawContours(skeleton, [contour], -1, 255, 1)

    return skeleton

def extract_grains_from_skeleton(
    skeleton,
    pad=0,
    dilation_kernel=(3, 3),
    contour_shift=(0, 0),
    start_id=1,
    mark_twinning=False
):
    """
    Extract Grain objects from a skeleton image.
    """
    if pad > 0:
        skeleton = np.pad(skeleton, pad, constant_values=1)

    kernel = np.ones(dilation_kernel, np.uint8)
    dilated = cv2.dilate(skeleton, kernel, iterations=1)

    labels = label(~dilated.astype(bool))
    contours = find_contours(labels, 0.5)

    grains = []

    for i, contour in enumerate(contours):
        contour = np.round(contour).astype(int)
        contour = np.flip(contour, axis=1)

        # Undo padding if needed
        contour[:, 0] -= contour_shift[0]
        contour[:, 1] -= contour_shift[1]

        # Interior points
        points_inside = Grain_functions.get_integer_points_inside_contour(contour)
        points = np.asarray(points_inside, dtype=np.int32)

        if len(points) == 0:
            continue

        center = points.mean(axis=0)

        gr = Grain_functions.Grain(
            points,
            contour,
            tuple(center),
            len(points),
            1,
            start_id + i
        )

        if mark_twinning:
            gr.is_twinning(True)

        grains.append(gr)

    return grains

def skeletonize_binary(binary_img):
    """
    Skeletonize a binary image and return uint8 (0 or 255)
    """
    ske = skeletonize(binary_img.astype(bool))
    return (ske.astype(np.uint8) * 255)

def delete_small_twins(Grains):
    index = []

    for i, gr in enumerate(Grains):
        if gr.IsTwin == True:
            if gr.size <= 5:
                index.append(i)

    for m in index:
        Grains.pop(m)

    return Grains

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
    if not files: return grain_stats, np.array([])

    # Get dimensions from the first image
    sample_img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    h, w = sample_img.shape

    num_grains = len(grain_stats)
    num_images = len(files)
    gray_mean_array = np.zeros((num_grains, num_images), dtype=np.float32)

    # Precompute indices with safety clipping
    grain_pixels = []
    for grain in grain_stats:
        pixels = np.asarray(grain.PixelList, dtype=np.int32)
        # Assuming PixelList is [x, y]
        x_idx = np.clip(pixels[:, 0], 0, w - 1)
        y_idx = np.clip(pixels[:, 1], 0, h - 1)
        grain_pixels.append((y_idx, x_idx))

    # Process images
    for img_idx, fname in enumerate(tqdm(files, desc="Calculating Gray Means")):
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        for i, (y, x) in enumerate(grain_pixels):
            # Rapid vectorized mean for this grain
            gray_mean_array[i, img_idx] = np.mean(image[y, x])

    for i, grain in enumerate(grain_stats):
        # Update grain with the new mean array
        grain.GrayMean = gray_mean_array[i]
        # Calculate centroid if needed
        y_idx, x_idx = grain_pixels[i]
        grain.Position = (np.mean(x_idx), np.mean(y_idx))

    return grain_stats, gray_mean_array.mean(axis=0)

def select_orientation_folder(original_path, current_directory):
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
    """
    Generate multiple pseudocolour images using different samples orientation to improve twins detection

    Args:
    orientation_path (str): Folder where all the images are saved for the different orientation.
    random_flat (int 0 or 1): Pseudocolour images can either be generated randomly or following the patterns below.

    Returns:
    pseudo_imgs (list of np.array): 3 images generated combining grayscale images at different orientation.
    SizeIm (tuples of int): Image size.
    """

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


def Peaks_Optimized(Grainstats, initial_s=50, s_increment=50, max_threshold=254, target_mean=122.5):
    """
    Estimate peak orientations for grains using a smoothing spline.
    If the maximum amplitude exceeds `max_threshold`, the smoothing parameter `s` is increased by `s_increment`.
    The loop runs for a maximum of 4 iterations.

    Parameters:
    - Grainstats: list of objects with attributes GrayMean and method set_orientation(azimuth, inclination)
    - initial_s: starting smoothing factor for UnivariateSpline
    - s_increment: how much to increase smoothing factor if threshold exceeded
    - max_threshold: amplitude threshold to adjust smoothing

    Returns:
    - Grainstats with orientations set
    """
    MAX_AMP_FIXED = 255
    # x_input: 0, 10, 20... 350 (36 points)
    x_input = np.arange(0, 360, 10)
    # x_fine: 0.1 degree resolution (3600 points)
    x_fine = np.arange(0, 360, 0.1)
    max_iter = 4

    for grain in Grainstats:
        # 1. Pre-processing & Vectorized Shifting
        gray = np.array(grain.GrayMean)
        gray = gray + (target_mean - np.mean(gray))

        # Ensure data length matches x_input (36 points)
        if gray.size > 36:
            gray = gray[:36]
        elif gray.size < 36:
            # Pad with mean if data is unexpectedly short
            gray = np.pad(gray, (0, 36 - gray.size), mode='mean')

        # 2. Spline Fitting
        # Tile twice to handle circularity/wrap-around properly
        gray_extended = np.tile(gray, 2)
        x_ext = np.arange(0, 720, 10)
        iter = 0
        ok = 0
        s_value = initial_s

        while ok == 0 and iter <= max_iter:

            spline = UnivariateSpline(x_ext, gray_extended, s=s_value)
            smoothed = spline(x_fine)  # We only evaluate the first 360 degrees

            # 3. Peak Detection
            s_min, s_max = smoothed.min(), smoothed.max()
            height_thresh = s_min + 0.6 * (s_max - s_min)

            peaks, info = find_peaks(smoothed, height=height_thresh)
            pks = info.get('peak_heights', [])

            # Logic for finding the primary location
            if pks is not None and len(pks) > 0:
                # Get index of the highest peak
                best_idx = np.argmax(pks)
                best_loc = peaks[best_idx] * 0.1  # Convert index to degrees
                amp_val = pks[best_idx] - s_min
                ok = 1
            else:
                # Fallback: if no peak found, use the global max
                shifted = np.roll(smoothed, 90)
                peaks, info = find_peaks(shifted, height=height_thresh)
                pks = info.get('peak_heights', [])
                if pks is not None and len(pks) > 0:
                    best_idx = np.argmax(pks)
                    best_loc = peaks[best_idx] * 0.1  # Convert index to degrees
                    amp_val = pks[best_idx] - s_min
                    ok = 1
                else:
                    s_value = s_value + 50
                    iter = iter + 1

            # 4. Final Orientation Assignment
            # Using your specific formulas
        if iter == max_iter:
            peaks, info = find_peaks(smoothed)
            pks = info.get('peak_heights', [])
            best_idx = np.argmax(pks)
            best_loc = peaks[best_idx] * 0.1  # Convert index to degrees
            amp_val = pks[best_idx] - s_min
            if amp_val > MAX_AMP_FIXED:
                amp_val = MAX_AMP_FIXED
        inclination = 0 if amp_val <= 0 else (amp_val * 90 / MAX_AMP_FIXED)
        azimuth = (best_loc - 45) % 180

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


# --- Step 1: Generate Skeleton from Pixel List ---
def get_skeleton(grain, image_shape):
    if len(grain.SkeletonCoord) == 0:
        mask = np.zeros(image_shape, dtype=bool)
        twin_pixels = np.array(grain.PixelList)
        mask[twin_pixels[:, 0], twin_pixels[:, 1]] = 1
        skeleton = skeletonize(mask)
    else:
        mask = np.zeros(image_shape, dtype=bool)
        twin_pixels = np.array(grain.SkeletonCoord)
        mask[twin_pixels[:, 0], twin_pixels[:, 1]] = 1
        skeleton = skeletonize(mask)
    return skeleton


# --- Step 2: Extract branches from skeleton ---
def get_branches(skeleton):
    skel = Skeleton(skeleton)
    branches = summarize(skel, separator='_')
    return skel, branches


# --- Step 3: Get branch endpoints ---
def get_branch_endpoints_centroid(skel, branches, grain):
    endpoints = []
    centroid = []
    for _, row in branches.iterrows():
        src = row[1]
        dst = row[2]
        pt1 = skel.coordinates[int(src)]
        pt2 = skel.coordinates[int(dst)]
        endpoints.append((pt1, pt2))
    centroid_1 = np.transpose(grain.Centroid)
    centroid_int = centroid_1.astype(np.int64)

    centroid.append(centroid_int)

    return endpoints, centroid_int


# --- Step 4: Get centroids of neighbour grains ---
def compute_centroids(grain_ids, grains):
    centroids = {}
    for gr in grains:
        if gr.ID in grain_ids:
            pixels = np.array(gr.PixelList)
            centroids[gr.ID] = np.mean(pixels, axis=0)
    return centroids

def check_twins(Grains, list_grains):
    for grain in Grains:

        if grain.ID in list_grains:

            if grain.IsTwin == True:
                list_grains.remove(grain.ID)

    return list_grains

def get_optimized_neighbours(Grainstats, image_input):
    # Detect if image or shape was passed
    if hasattr(image_input, 'shape'):
        shape = image_input.shape[:2]
    else:
        shape = image_input[:2]

    z_final = np.zeros(shape, dtype=np.int32)
    rows_max, cols_max = shape[0] - 1, shape[1] - 1

    for grain in Grainstats:
        pixels = np.asarray(grain.PixelList, dtype=int)

        # Consistent Indexing: pixels[:, 1] is Y (Rows), pixels[:, 0] is X (Cols)
        # We clip to prevent "Index 509 is out of bounds for size 509"
        y_idx = np.clip(pixels[:, 1], 0, rows_max)
        x_idx = np.clip(pixels[:, 0], 0, cols_max)

        z_final[y_idx, x_idx] = grain.ID

    # Adjacency logic (Grid Shifting)
    h_neighbors = np.stack((z_final[:, :-1], z_final[:, 1:]), axis=-1).reshape(-1, 2)
    v_neighbors = np.stack((z_final[:-1, :], z_final[1:, :]), axis=-1).reshape(-1, 2)

    all_pairs = np.vstack([h_neighbors, v_neighbors])
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
    all_pairs = all_pairs[(all_pairs[:, 0] != 0) & (all_pairs[:, 1] != 0)]

    all_pairs.sort(axis=1)
    unique_pairs = np.unique(all_pairs, axis=0)

    adj_dict = {grain.ID: set() for grain in Grainstats}
    for id1, id2 in unique_pairs:
        if id1 in adj_dict: adj_dict[id1].add(id2)
        if id2 in adj_dict: adj_dict[id2].add(id1)

    for grain in tqdm(Grainstats, desc="Updating Neighbors"):
        if grain.IsTwin or grain.HaveFriends:
            neighbor_ids = adj_dict.get(grain.ID, set())
            final_neigh = check_twins(Grainstats, neighbor_ids)
            grain.set_neighbours(list(final_neigh))
        else:
            grain.Neighbours = []

    return Grainstats


def find_parents_separate_twins(grains, image_shape, Zfinal, Average_Size, skeleton_grains, background=None):
    for i, grain in enumerate(grains):

        if grain.IsTwin:

            skeleton = get_skeleton(grain, image_shape)

            skel, branches = get_branches(skeleton)
            endpoints, centroido = get_branch_endpoints_centroid(skel, branches, grain)
            neighbour_ids = grain.Neighbours
            centroids = compute_centroids(neighbour_ids, grains)
            centroids2 = copy.deepcopy(centroids)
            branch_results = []

            neighs_left_ID = []
            neighs_right_ID = []

            Parents = False
            neighs_left = []
            neighs_right = []
            neighs_left_length = []
            neighs_right_length = []
            points = endpoints[0]

            for nid, centroid in centroids.items():
                if nid == 70:
                    kk = 1
                if is_projection_inside_segment(np.array(points[0]), np.array(points[1]), centroid):
                    length, left = is_left_or_right(np.array(points[0]), np.array(points[1]), centroid)
                    if left:
                        neighs_left_length.append(length)
                        neighs_left_ID.append(nid)
                        neighs_left.append(find_grain_by_ID(grains, nid))

                    else:
                        neighs_right_length.append(length)
                        neighs_right_ID.append(nid)
                        neighs_right.append(find_grain_by_ID(grains, nid))

                if nid in centroids2:
                    del centroids2[nid]

            if (len(neighs_left) == 1 and len(neighs_right) == 1):
                Azimuth_P, Incli_P, Parents = check_friends(neighs_left[0], neighs_right[0])
                # plot_projection_inside(grain, image_shape, neighs_left[0], neighs_right[0])
                if Parents == True:
                    grain.Dad = neighs_left[0].ID
                    grain.Mum = neighs_right[0].ID
            elif (len(neighs_left) == 0 and len(neighs_right) == 1) or (
                    len(neighs_left) == 1 and len(neighs_right) == 0):
                if len(neighs_right) == 1:
                    Azimuth_P = neighs_right[0].Azimuth
                    Incli_P = neighs_right[0].Inclination
                    grain.Mum = neighs_right[0].ID
                    Parents = True
                if len(neighs_left) == 1:
                    Azimuth_P = neighs_left[0].Azimuth
                    Incli_P = neighs_left[0].Inclination
                    grain.Dad = neighs_left[0].ID
                    Parents = True

            elif (len(neighs_left) == len(neighs_right)) and (len(neighs_right) > 1):
                # elif (len(neighs_left) == len(neighs_right)) or (len(neighs_right) > 1):
                neighs_left = [val for _, val in sorted(zip(neighs_left_length, neighs_left))]
                neighs_right = [val for _, val in sorted(zip(neighs_right_length, neighs_right))]
                neighs_left_length = [val for _, val in sorted(zip(neighs_left_length, neighs_left_length))]
                neighs_right_length = [val for _, val in sorted(zip(neighs_right_length, neighs_right_length))]

                Azimuth_P, Incli_P, Parents = check_friends(neighs_left[0], neighs_right[0])
                if Parents == True:
                    the_goat, new_granulo = separate_twin(grain, neighs_left[0], neighs_right[0], image_shape,
                                                          find_max_ID(grains), skeleton_grains)
                    if new_granulo:
                        for gr in new_granulo:
                            # skeleton = get_skeleton(gr, image_shape)
                            # skel, branches = get_branches(skeleton)
                            # gr.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)
                            grains.append(gr)
                    # skeleton = get_skeleton(the_goat, image_shape)
                    # the_goat.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)
                    grains[i] = the_goat



            else:
                totot = 3

    return grains


def measure_contour_length(contour):
    length = 0

    for i in range(1, len(contour)):
        x1 = int(contour[i - 1, 0])
        y1 = int(contour[i - 1, 1])

        x2 = int(contour[i, 0])
        y2 = int(contour[i, 1])

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        length = length + distance

    x1 = int(contour[0, 0])
    y1 = int(contour[0, 1])

    x2 = int(contour[len(contour) - 1, 0])
    y2 = int(contour[len(contour) - 1, 1])

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return length


def gray_mean_twin(grain_stats, folder, brightness, contrast, plot=False, correction_method=False):
    files = sorted(glob.glob(os.path.join(folder, '*.png')), key=extract_number2)
    num_grains = len(grain_stats)
    num_images = len(files)

    gray_mean_array = np.zeros((num_grains, num_images))
    grain_positions = np.zeros((num_grains, 2))
    count = np.zeros(num_grains)
    images = []

    # --- Preprocess all images ---
    for m in range(num_images):
        image = cv2.imread(files[m], cv2.IMREAD_GRAYSCALE)

        # APPLY BRIGHTNESS & CONTRAST
        img_bc = adjust_brightness_contrast(
            image,
            brightness_pct=brightness,
            contrast_pct=contrast
        )
        if correction_method == True:
            img_clahe = apply_clahe(img_bc)
            img_contrast = adjust_contrast(img_clahe)
            img_normalize1 = normalize_images_all(img_contrast)
            img_normalize = np.transpose(img_normalize1[0])
        # img_normalize = img_bc

        else:
            img_normalize = img_bc

        images.append(img_normalize)

    images_final = images
    # --- Initialize variable for manual points ---
    manual_pts = None

    # --- Main loop over images ---
    for img_idx in range(num_images):
        image = images_final[img_idx]

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Image {img_idx} - {os.path.basename(files[img_idx])}")
            ax.axis('off')

        for i, grain in enumerate(grain_stats):
            if grain.IsTwin:

                # else:
                pixels = np.array(grain.SkeletonCoord)

                # Compute gray mean
                y = pixels[:, 1].astype(int)
                x = pixels[:, 0].astype(int)
                gray_values = image[y, x]
                gray_mean_array[i, img_idx] = np.mean(gray_values)
                grain_positions[i] = [np.mean(x), np.mean(y)]

                if plot:
                    color = 'lime' if grain.ID == 132 else 'red'
                    ax.scatter(x, y, s=2, color=color)

            else:
                pixels = np.array(grain.PixelList)
                y = pixels[:, 1].astype(int)
                x = pixels[:, 0].astype(int)
                gray_values = image[y, x]
                gray_mean_array[i, img_idx] = np.mean(gray_values)
                grain_positions[i] = [np.mean(x), np.mean(y)]

        if plot:
            plt.show()

    # --- Update Grainstats with new data ---
    for i, grain in enumerate(grain_stats):
        grain.Position = grain_positions[i]
        grain.GrayMean = gray_mean_array[i]
        count[i] = np.sum(grain.GrayMean < 10)

    result = np.mean(gray_mean_array, axis=0)
    return grain_stats, result

def adjust_brightness_contrast(image, brightness_pct, contrast_pct):
    """
    Adjusts brightness and contrast of a grayscale image by percentages.

    Parameters:
    - image: Input grayscale image (numpy array)
    - brightness_pct: Percentage to increase/decrease brightness (-100 to 100)
    - contrast_pct: Percentage to increase/decrease contrast (-100 to 100)

    Returns:
    - Adjusted image (uint8)
    """
    # Convert percentages to alpha and beta
    alpha = 1 + (contrast_pct / 100.0)  # Contrast factor
    beta = (brightness_pct / 100.0) * 255  # Brightness offset

    # Apply adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# %% MAIN FUNCTION
def orientation_and_classification(orientation_path, grains_image_path, twins_image_path, image_name):

    #Initialisation
    print("Grains Process Starting...")

    Grains = []
    path_gr = grains_image_path
    path_tw = twins_image_path
    img_study = image_name
    grains = cv2.imread(grains_image_path, cv2.COLOR_BGR2GRAY)
    nothing, size = Grain_functions.image_size(grains_image_path)
    twins = cv2.imread(twins_image_path, cv2.COLOR_BGR2GRAY)

    skeleton_grains = skeletonize_binary(~grains)
    skeleton_twins = skeletonize_binary(~twins)

    # ======================================================
    # 2. Extract twin grains
    # ======================================================

    kernel_5 = np.ones((5, 5), np.uint8)
    dilated_twins = cv2.dilate(skeleton_twins, kernel_5, iterations=1)

    labels_twins = label(dilated_twins.astype(bool))
    twin_contours = find_contours(labels_twins, 0.5)

    Twins = []
    twin_contours_int = []

    for i, contour in enumerate(twin_contours):
        contour = np.round(contour).astype(int)
        contour = np.flip(contour, axis=1)

        twin_contours_int.append(contour)

        points_inside = Grain_functions.get_integer_points_inside_contour(contour)
        points = np.asarray(points_inside, dtype=np.int32)

        if len(points) == 0:
            continue

        center = points.mean(axis=0)

        gr = Grain_functions.Grain(
            points,
            contour,
            tuple(center),
            len(points),
            1,
            i + 1
        )

        gr.is_twinning(True)
        Twins.append(gr)

    # ======================================================
    # 3. Remove twins from grain skeleton
    # ======================================================

    skeleton_grains = subtract_contours_from_skeleton(
        skeleton_grains,
        twin_contours_int
    )

    skeleton_grains = skeletonize_binary(skeleton_grains)

    # ======================================================
    # 4. Extract grains
    # ======================================================

    Grains = extract_grains_from_skeleton(
        skeleton=skeleton_grains,
        pad=1,
        dilation_kernel=(3, 3),
        contour_shift=(1, 1),
        start_id=1,
        mark_twinning=False
    )

    # ======================================================
    # 5. Spatially indexed grain–twin intersection
    # ======================================================

    # Build shapely geometries
    twin_polys = [poly_line(tw) for tw in Twins]
    grain_polys = [poly_line(gr) for gr in Grains]

    # Build spatial index
    twin_tree = STRtree(twin_polys)

    for gr, grain_poly in zip(Grains, grain_polys):

        grain_area = grain_poly.area
        if grain_area == 0:
            continue

        # 1. Query returns indices of twin_polys whose bounding boxes overlap grain_poly
        candidate_indices = twin_tree.query(grain_poly)

        for idx in candidate_indices:
            # 2. Get the twin object and the geometry using the index
            twin_obj = Twins[idx]
            t_poly_geom = twin_polys[idx]

            twin_area = t_poly_geom.area
            if twin_area == 0:
                continue

            # 3. Precise geometric check
            if not grain_poly.intersects(t_poly_geom):
                continue

            intersection = grain_poly.intersection(t_poly_geom)
            if intersection.is_empty:
                continue

            overlap_area = intersection.area
            twinning_ratio = overlap_area / grain_area

            gr.add_twinning_area1(twinning_ratio)

            # Note: Usually in EBSD/microscopy, exact 1.0 matches are rare
            # due to pixelation, so you might consider a threshold like > 0.95
            if twinning_ratio >= 0.99:
                gr.is_twinning(True)

    final_orientation = orientation_path
    check_grains(Grains)

    Grains, result = gray_mean(Grains, final_orientation)

    Grains = Peaks_Optimized(Grains)
    # plot_grains_contour(Grains3, grains.shape)
    check_peaks(Grains)
    grano = grains
    Grains = get_optimized_neighbours(Grains, grano.shape)

    check_grains(Grains)

    id_twins = []
    Error_Angle = 6
    A = []
    for gr in Grains:
        if gr.IsTwin == True:
            id_twins.append(gr.ID)

            for m in range(len(gr.Neighbours)):
                studied_grain = Grains[gr.Neighbours[m] - 1]
                phi1 = studied_grain.Azimuth
                Phi = studied_grain.Inclination
                id1 = studied_grain.ID

                for n in range(len(gr.Neighbours)):
                    if n != m:
                        A = []
                        phi1 = studied_grain.Azimuth
                        Phi = studied_grain.Inclination
                        id1 = studied_grain.ID
                        studied_grain2 = Grains[gr.Neighbours[n] - 1]
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


    Grains = get_optimized_neighbours(Grains,  grano.shape)

    check_grains(Grains)
    Grains, result = gray_mean(Grains, final_orientation)

    Grains = Peaks_Optimized(Grains)
    overlapping_grains, ID_grains = Grain_functions.find_overlapping_grains(Grains)

    for i, grains_couple in tqdm(enumerate(overlapping_grains), total=len(overlapping_grains),
                                 desc="Processing grains"):
        if any(grain.AtRisk for grain in grains_couple):
            Grain_functions.remove_overlapping_pixels_ATRISK(grains_couple[0], grains_couple[1], size)

    new_grains = []
    for gr in Grains:
        if gr.AtRisk and (len(gr.ContourPoints) == 0 or len(gr.PixelList) == 0):
            print(f"Removed grain ID {gr.ID} due to empty ContourPoints or PixelList.")
            continue
        new_grains.append(gr)

    Grains6 = new_grains
    for i, gr in enumerate(Grains6):
        if gr.AtRisk == True:

            # gr.PixelList = np.flip(gr.PixelList, axis=1)
            gr.ContourPoints = np.flip(gr.ContourPoints, axis=1)

    Grains7 = get_optimized_neighbours(Grains6, grano.shape)
    grain_matrix2 = np.zeros(np.flip(grano.shape), dtype=np.int32)
    mask = np.zeros(np.flip(grano.shape), dtype=np.uint8)
    for grain in Grains7:

        for (x, y) in grain.PixelList:
            try:
                grain_matrix2[x, y] = grain.ID
                mask[x, y] = 1
            except IndexError as e:
                print(f"IndexError: Grain ID {grain.ID} caused an out-of-bounds error at position ({x}, {y})")
                continue
    Grains7 = sorted(Grains7, key=lambda g: g.confidence, reverse=True)

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

    kernel = np.ones((3, 3), np.uint8)  # Or larger if you want more dilation
    # plot_grains(Grains7, grains.shape)
    for grain in Grains7:
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
    Grains7 = [gr for gr in Grains7 if len(gr.PixelList) > min_size]

    Grains7 = get_optimized_neighbours(Grains7, grano.shape)
    Grains10 = copy.deepcopy(Grains7)
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

    Grains10 = get_optimized_neighbours(Grains10, grano.shape)

    Grains10, result = gray_mean_twin(Grains10, final_orientation, 0, 0)
    Grains10 = Peaks_Optimized(Grains10)
    Grains10 = [gr for gr in Grains10 if gr.size > min_size]
    Zfinal = np.zeros(np.flip(grano.shape), dtype=int)

    for gr in Grains10:
        pixel_list = np.array(gr.PixelList)
        Zfinal[pixel_list[:, 0], pixel_list[:, 1]] = gr.ID

    Grains10 = find_parents_separate_twins(Grains10, np.flip(grano.shape), Zfinal, Average_Size, skeleton_grains, background=grano)
    Grains10 = decompose_twins_grains_2(Grains10, grano, Average_length, Std_length)

    Grains10, result = gray_mean_twin(Grains10, final_orientation, 0, 0)
    Grains10 = Peaks_Optimized(Grains10)

    for gr in Grains10:
        if gr.IsTwin:
            gr, type_error = grain_twin_analysis(gr, Grains10, np.flip(grano.shape), Zfinal, Average_Size,
                                                 background=grano)

    for gr in Grains10:
        if gr.IsTwin:
            if gr.TwinType == "Tension" or gr.TwinType == "Compression":
                if gr.Mum:
                    index = find_grain_by_ID_index(Grains10, gr.Mum)
                    Grains10[index].IsParents = True
                if gr.Dad:
                    index = find_grain_by_ID_index(Grains10, gr.Dad)
                    Grains10[index].IsParents = True

    final_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for grain in Grains10:
        # 1. Determine color and thickness based on twin status
        if grain.IsTwin:
            if grain.TwinType == "Tension":
                color = (50, 205, 50)  # Green
                thickness = 2
            elif grain.TwinType == "Compression":
                color = (255, 0, 0)  # Blue (Note: OpenCV is BGR, so 255,0,0 is BLUE)
                thickness = 2
            else:
                color = (0, 0, 255)  # Red (0,0,255 is RED in BGR)
                thickness = 2
        else:
            color = (0, 0, 0)  # Black for normal grains
            thickness = 1

        # 2. Draw directly onto the final image
        try:
            cv2.drawContours(
                final_image,
                [grain.DilatedContourPoints],
                -1,
                color,
                thickness
            )
        except Exception as e:
            print(f"Drawing error for Grain {grain.ID}: {e}")

    # Apply final transformations
    final_image = np.flipud(np.rot90(final_image))
    colormap_path = r'D:\PLM_ML_Twins_Classification\files\Colormap\four_w.png'
    mode = 3
    ColorMap = Grain_Orientation.grain_orientation(Grains10, mode, dilated_contours, colormap_path)

    FinalPlot = ColorMap.copy()

    # 3. Apply final transformations
    FinalPlot_rotated = cv2.rotate(FinalPlot, cv2.ROTATE_90_CLOCKWISE)
    FinalPlot_transformed = cv2.flip(FinalPlot_rotated, 1)
    FinalPlot_rgb = cv2.cvtColor(FinalPlot_transformed, cv2.COLOR_BGR2RGB)

    # Convert FinalPlot_rgb back to BGR for OpenCV display
    FinalPlot_bgr = cv2.cvtColor(FinalPlot_rgb, cv2.COLOR_RGB2BGR)

    return final_image, FinalPlot_rgb