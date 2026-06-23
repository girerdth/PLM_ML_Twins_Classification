# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 10:33:50 2026

Author: Thomas Girerd

Description:
------------
This script performs grain segmentation and morphological analysis on
microstructural images. It includes tools for contour extraction,
grain reconstruction, skeleton analysis, twin decomposition, overlap
handling, and geometric characterization of grains.

Main libraries:
- OpenCV for image processing
- NumPy for numerical operations
- scikit-image for morphology and connected components
- skan for skeleton graph analysis
"""

# ============================================================
# Python Packages
# ============================================================

from ultralytics import YOLO
import cv2
import os
import numpy as np
from skimage.measure import regionprops, label
import copy
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label, find_contours
import math
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from collections import defaultdict
from joblib import Parallel, delayed
from skimage.morphology import skeletonize, thin
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize

# ============================================================
# Utility Functions
# ============================================================

def get_integer_points_inside_contour(contour):
    """
    Returns all integer coordinate points located inside a contour.

    Parameters
    ----------
    contour : ndarray
        Contour coordinates defining a polygon.

    Returns
    -------
    points_inside_contour : list
        List of (x, y) integer coordinates inside the contour.

    Purpose
    -------
    Used to reconstruct all pixels belonging to a segmented grain.
    """
    x_min = np.min(contour[:, 0])
    x_max = np.max(contour[:, 0])
    y_min = np.min(contour[:, 1])
    y_max = np.max(contour[:, 1])
    
    points_inside_contour = []

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                points_inside_contour.append((x, y))
    
    
    return points_inside_contour


def generate_grain_matrix(seg_grains, image_size):
    """
    Creates an image-sized matrix where each pixel stores grain ID.

    Parameters
    ----------
    seg_grains : list
        List of Grain objects.

    image_size : tuple
        Size of image (height, width).

    Returns
    -------
    grain_matrix : ndarray
        Matrix containing grain IDs.

    mask : ndarray
        Binary mask showing occupied grain pixels.

    Purpose
    -------
    Converts grain pixel lists into matrix representation for
    later morphological operations.
    """
    grain_matrix = np.zeros(image_size, dtype=np.int32)
    mask = np.zeros(image_size, dtype=np.uint8)
    for grain in seg_grains:
        for (x, y) in grain.PixelList:
            try:
                grain_matrix[x, y] = grain.ID
                mask[x, y] = 1
            except IndexError:
                continue
    return grain_matrix, mask

def fill_contour_gaps(seg_grains, grain_matrix, mask, image_size):
    """
    Expands grain contours slightly in order to fill segmentation gaps.

    Parameters
    ----------
    seg_grains : list
        List of grain objects.

    grain_matrix : ndarray
        Grain ID matrix.

    mask : ndarray
        Binary mask of occupied pixels.

    image_size : tuple
        Image dimensions.

    Returns
    -------
    grain_matrix2 : ndarray
        Updated grain matrix with contour gaps filled.

    Purpose
    -------
    Fixes small discontinuities along predicted grain boundaries.
    """
    grain_matrix2 = grain_matrix.copy()
    sorted_grains = sorted(seg_grains, key=lambda g: g.confidence, reverse=True)

    for grain in sorted_grains:
        mask2 = np.zeros(image_size, dtype=np.uint8)
        cv2.drawContours(mask2, [grain.ContourPoints], -1, (1), thickness=1)
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask2 = cv2.dilate(mask2, kernel, iterations=1)
        difference_mask = dilated_mask2 - mask
        grain_matrix2[difference_mask == 1] = grain.ID

    return grain_matrix2

def add_missing_regions(grain_matrix2, existing_grains, image_size):
    """
    Finds empty image regions and assigns them new grain IDs.

    Parameters
    ----------
    grain_matrix2 : ndarray
        Existing grain matrix.

    existing_grains : list
        Current grain objects.

    image_size : tuple
        Image dimensions.

    Returns
    -------
    grain_matrix2 : ndarray
        Updated matrix with missing regions filled.

    Purpose
    -------
    Ensures every image pixel belongs to some grain.
    """

    zero_mask = (grain_matrix2 == 0).astype(np.uint8)
    labeled_zeros, _ = label(zero_mask, return_num=True)
    max_id = max(grain.ID for grain in existing_grains)
    new_id = max_id + 1

    for region in regionprops(labeled_zeros):
        if region.area > 0:
            coords = region.coords
            for (x, y) in coords:
                try:
                    grain_matrix2[x, y] = new_id
                except IndexError:
                    continue
            new_id += 1

    return grain_matrix2

def create_contour_mask(grain_matrix2):
    """
    Creates skeletonized contour map from grain matrix.

    Parameters
    ----------
    grain_matrix2 : ndarray
        Grain ID matrix.

    Returns
    -------
    dilated : ndarray
        Skeletonized contour mask.

    Purpose
    -------
    Extracts grain boundaries for topology/skeleton analysis.
    """

    kernel = np.ones((3, 3), np.uint8)
    smoothed_mask = cv2.morphologyEx(grain_matrix2.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

    gradient_x = cv2.Sobel(smoothed_mask, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed_mask, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    contour_mask = (gradient_magnitude > 0).astype(np.uint8)
    skeletonized = skeletonize(contour_mask.astype(bool)).astype(np.uint8)

    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(skeletonized, kernel_dilate, iterations=1)

    return dilated

class Grain:
    """
    Stores all information related to one segmented grain.

    Attributes
    ----------
    PixelList : ndarray
        Pixel coordinates belonging to grain.

    ContourPoints : ndarray
        Grain contour coordinates.

    Centroid : tuple
        Grain center coordinates.

    size : int
        Number of pixels in grain.

    confidence : float
        Segmentation confidence score.

    Purpose
    -------
    Main object used throughout the analysis pipeline.
    """
    def __init__(self, pixel_list, contour_points, centroid, size, confidence, ID, is_twin=False, azimuth=None, inclination=None):
        self.PixelList     = pixel_list
        self.ContourPoints = contour_points
        self.Centroid      = centroid
        self.size          = size
        self.confidence    = confidence
        self.ID            = ID
        self.ID2           = 0
        self.IsTwin        = is_twin
        self.Azimuth       = azimuth
        self.Inclination   = inclination
        self.ContourLength = 0
        
        def define_ellipsoid(pixel_list, centroid, scale='sqrt2'):
            """
            Computes ellipse approximation from grain pixels.

            Returns
            -------
            a : float
                Major axis length

            b : float
                Minor axis length

            tau : float
                Orientation angle
            """
            # central moments
            dx = pixel_list[:, 0] - centroid[0]
            dy = pixel_list[:, 1] - centroid[1]
            u20 = np.mean(dx*dx)
            u02 = np.mean(dy*dy)
            u11 = np.mean(dx*dy)
        
            # orientation (stable form)
            tau = 0.5 * np.arctan2(2*u11, (u20 - u02))
        
            # eigenvalues (variances along principal axes)
            trace = u20 + u02
            diff  = u20 - u02
            rad   = np.sqrt(diff*diff + 4*u11*u11)
            lam1  = 0.5*(trace + rad)
            lam2  = 0.5*(trace - rad)
            lam1  = max(lam1, 0.0)
            lam2  = max(lam2, 0.0)
        
            if scale == 'sigma1':
                a = np.sqrt(lam1)
                b = np.sqrt(lam2)
            else:  # 'sqrt2' default
                a = np.sqrt(2*lam1)
                b = np.sqrt(2*lam2)
        

            return a, b, tau

        
        self.l1, self.l2, self.tau = define_ellipsoid(pixel_list,centroid)
        

        # Twinning areas for various analyses
        self.twinning_area1 = 0.0
        self.twinning_area2 = 0.0
        self.twinning_area3 = 0.0

        # New attributes for grayscale analysis
        self.GrayMean  = []  # To be filled with a list/array of gray mean values
        self.Position  = None  # (x, y) center of mass or average position
        self.GrayCount = 0  # Number of images with gray mean below threshold
        
        self.Neighbours = []
        self.Friends = []
        self.SkeletonCoord = []
        self.HaveFriends = False
        self.Dad = []
        self.Mum = []
        self.IsParents = False
        self.TwinType = "None"
        self.AtRisk = False
        
        self.SideNeighbours = []
        self.ExtremityNeighbours = []
        self.MisOrientation = []
           
       
    def update_ellipsoid(self, scale='sqrt2'):
        """
        Recomputes ellipse approximation after grain geometry changes.

        Parameters
        ----------
        scale : str
            Controls ellipse scaling.

            'sqrt2' -> semi-axis = sqrt(2 * eigenvalue)
            'sigma1' -> semi-axis = sqrt(eigenvalue)

        Purpose
        -------
        Called whenever PixelList changes (for example after overlap
        correction or grain decomposition).
        """
        # central moments
        pixel_list = np.array(self.PixelList)
        centroid = np.array(self.Centroid)
        
        
        dx = pixel_list[:, 0] - centroid[0]
        dy = pixel_list[:, 1] - centroid[1]
        u20 = np.mean(dx*dx)
        u02 = np.mean(dy*dy)
        u11 = np.mean(dx*dy)
    
        # orientation (stable form)
        tau = 0.5 * np.arctan2(2*u11, (u20 - u02))
    
        # eigenvalues (variances along principal axes)
        trace = u20 + u02
        diff  = u20 - u02
        rad   = np.sqrt(diff*diff + 4*u11*u11)
        lam1  = 0.5*(trace + rad)
        lam2  = 0.5*(trace - rad)
        lam1  = max(lam1, 0.0)
    
        if scale == 'sigma1':
            a = np.sqrt(lam1)
            b = np.sqrt(lam2)
        else:  # 'sqrt2' default
            a = np.sqrt(2*lam1)
            b = np.sqrt(2*lam2)
    

        self.l1 = a
        self.l2 = b
        self.tau = tau
        
        return 

    def add_twinning_area1(self, area):
        self.twinning_area1 += area
        return self.twinning_area1

    def is_twinning(self, is_twin):
        """
        Sets whether grain is identified as twin.

        Parameters
        ----------
        is_twin : bool
            Twin classification result.
        """
        self.IsTwin = is_twin
        return self.IsTwin

    def set_orientation(self, azimuth, inclination):
        """
        Stores crystallographic orientation values.

        Parameters
        ----------
        azimuth : float
            Grain azimuth angle.

        inclination : float
            Grain inclination angle.
        """
        self.Azimuth = azimuth
        self.Inclination = inclination

    def set_gray_analysis(self, gray_mean_values, position, threshold=10):
        """
        Stores grayscale analysis results.

        Parameters
        ----------
        gray_mean_values : list
            Mean grayscale intensity values.

        position : tuple
            Spatial grain position.

        threshold : int
            Intensity threshold.

        Purpose
        -------
        Counts number of grayscale values below threshold.
        Used for contrast-based material analysis.
        """
        self.GrayMean = gray_mean_values
        self.Position = position
        self.GrayCount = sum(val < threshold for val in gray_mean_values)
    
    def set_neighbours(self,neigh):
        """
        Stores neighboring grains.

        Parameters
        ----------
        neigh : list
            Neighbor grain IDs or objects.
        """
        self.Neighbours = neigh

    def add_friends(self,friends):
        """
        Adds related grains to friend list.

        Purpose
        -------
        Used when building grain connectivity graph.
        """
        
        self.HaveFriends = True
        toto = self.Friends
        toto.append(friends)
        self.Friends = toto  

# ============================================================
# Skeleton / Branch Analysis Utility Functions
# ============================================================

def find_pt(row1, row2):
    """
    Finds common endpoint shared by two skeleton branches.

    Parameters
    ----------
    row1, row2 : pandas Series
        Rows extracted from skeleton branch dataframe.

    Returns
    -------
    ndarray or None
        Shared point if branches intersect.
    """
    pt11 = np.array([int(row1["image_coord_src_0"]), int(row1["image_coord_src_1"])])
    pt21 = np.array([int(row1["image_coord_dst_0"]), int(row1["image_coord_dst_1"])])
         
    pt12 = np.array([int(row2["image_coord_src_0"]), int(row2["image_coord_src_1"])])
    pt22 = np.array([int(row2["image_coord_dst_0"]), int(row2["image_coord_dst_1"])])
    
    if np.array_equal(pt11, pt12) or np.array_equal(pt11, pt22):
        return pt11
    if np.array_equal(pt21, pt12) or np.array_equal(pt21, pt22):
        return pt21
    
    return None

def find_angle(row1, row2):
    """
    Computes angle between two skeleton branches.

    Parameters
    ----------
    row1, row2 : pandas Series
        Branch descriptions from skeleton dataframe.

    Returns
    -------
    angle : float
        Angle in degrees between branches.

    Purpose
    -------
    Used when decomposing junctions into separate twins.
    """
    pt11 = np.array([int(row1["image_coord_src_0"]), int(row1["image_coord_src_1"])])
    pt21 = np.array([int(row1["image_coord_dst_0"]), int(row1["image_coord_dst_1"])])
    pt12 = np.array([int(row2["image_coord_src_0"]), int(row2["image_coord_src_1"])])
    pt22 = np.array([int(row2["image_coord_dst_0"]), int(row2["image_coord_dst_1"])])

    A = (pt11 - pt21).astype(float)
    B = (pt12 - pt22).astype(float)

    # Normalize safely
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    if norm_A == 0 or norm_B == 0:
        return None  # invalid branch length

    A /= norm_A
    B /= norm_B

    # Dot product with clipping
    dot_product = np.clip(np.dot(A, B), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))

def safe_summarize_skeleton(skeleton_input):
    """
    Safely computes skeleton summary using skan.

    Parameters
    ----------
    skeleton_input : ndarray or Skeleton object

    Returns
    -------
    dataframe or None

    Purpose
    -------
    Prevents crashes when skeleton is too small or invalid.
    """
    # 1. Handle if it's already a Skeleton object
    if isinstance(skeleton_input, Skeleton):
        if skeleton_input.n_paths == 0:
            return None
        return summarize(skeleton_input, separator='_')

    # 2. If it's an image, check for minimum pixel count
    # A valid path needs at least 2 pixels to create an edge
    if skeleton_input is None or np.sum(skeleton_input) < 2:
        # print("⚠ Skeleton too small or empty (less than 2 pixels), skipping.")
        return None

    try:
        # 3. Create Skeleton object
        skel_obj = Skeleton(skeleton_input)

        # 4. Final check: does it actually have paths?
        if skel_obj.n_paths == 0:
            return None

        return summarize(skel_obj, separator='_')

    except (ValueError, ZeroDivisionError, Exception) as e:
        # This catches the SciPy 'index pointer' error specifically
        # print(f"⚠ Skan Error: {e}")
        return None
    
def find_grain_by_ID(grains, ID):
    """
    Finds grain object from list using grain ID.

    Parameters
    ----------
    grains : list
        List of Grain objects.

    ID : int
        Target grain ID.

    Returns
    -------
    Grain or None
    """
    for gr in grains:
        if gr.ID == ID:
            return gr
        
    return None

def find_grain_by_ID_index(grains, ID):
    """
    Finds index position of grain in list using grain ID.

    Parameters
    ----------
    grains : list
        List of Grain objects.

    ID : int
        Target grain ID.

    Returns
    -------
    int or None
        Index position in list.
    """
    for i, gr in enumerate(grains):
        if gr.ID == ID:
            return i
        
    return None

# ============================================================
# Twin Decomposition Functions
# ============================================================

def decompose_twins(skeleton, l1, std):
    """
    Decomposes a twin skeleton into multiple independent branches.

    Parameters
    ----------
    skeleton : ndarray
        Binary skeleton image representing one twin grain.

    l1 : float
        Characteristic grain length (currently unused here).

    std : float
        Standard deviation parameter (currently unused here).

    Returns
    -------
    skeleton : ndarray
        Updated skeleton after decomposition.

    branch_data : pandas.DataFrame
        Branch information extracted from skeleton.

    Purpose
    -------
    This function separates merged twin structures into individual
    linear twin branches.

    It solves two major problems:

    1. Curved twins
       → If branch length is much longer than straight-line distance,
         the branch likely contains multiple twins.

    2. Junction twins
       → If branches form Y-junctions or intersections, split them
         according to branch angles.

    Main workflow
    -------------
    Skeleton → detect abnormal branches → split branches →
    resolve junctions → return cleaned branches
    """

    # --------------------------------------------------------
    # Convert skeleton to graph representation
    # --------------------------------------------------------

    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
    branch_data = safe_summarize_skeleton(skeleton)
    no_change_count = 0
    THEFINALE = 0

    if branch_data is not None:
        # ----------------------------------------------------
        # Iterative decomposition loop
        # Continue until skeleton is fully decomposed
        # ----------------------------------------------------
        while THEFINALE == 0:
            final_coordinates = []

            # =================================================
            # PART 1 — Detect curved branches
            # =================================================
            #
            # If branch length is much longer than Euclidean
            # distance, the branch is not straight and probably
            # contains multiple twins merged together.
            #
            # Rule:
            # branch_distance > 1.25 × euclidean_distance
            #
            # =================================================

            for index, row in branch_data.iterrows():
                if row["branch_distance"] > row["euclidean_distance"] * 1.25:

                    ok = 0
                    skel_obj = Skeleton(skeleton)
                    branch_coords = skel_obj.path_coordinates(index)  # index of the current branch

                    # Create empty skeleton of the same shape
                    skeleton1 = np.zeros_like(skeleton, dtype=bool)

                    # Fill only the pixels of this branch
                    for x, y in branch_coords:
                        skeleton1[int(x), int(y)] = True

                    row_f = row
                    coordinates = copy.deepcopy(coordinates2)
                    ratio_d = 1.1

                    # ----------------------------------------
                    # Iteratively trim branch endpoints
                    # until branch becomes straight
                    # ----------------------------------------

                    while ok == 0:

                        branch_coords = coordinates
                        arr = np.column_stack(branch_coords)
                        if len(arr) == int(row_f["node_id_dst"]):
                            endpoint = arr[int(row_f["node_id_dst"]) - 1, :]
                        else:
                            endpoint = arr[int(row_f["node_id_dst"]), :]

                        x, y = endpoint
                        skeleton1[x, y] = 0
                        new_pixel_graph, coordinates = skeleton_to_csgraph(skeleton1)
                        new_branch_data = summarize(Skeleton(skeleton1), separator='_')

                        for ind, row2 in new_branch_data.iterrows():
                            row_f = row2
                            if row2["branch_distance"] <= row2["euclidean_distance"] * ratio_d:
                                branch_coords = coordinates
                                arr = np.column_stack(branch_coords)
                                endpoint = arr[int(row_f["node_id_dst"]), :]
                                x, y = endpoint
                                skeleton1[x, y] = 0
                                true_indices = np.argwhere(skeleton1)
                                ok = 1

                    # =================================================
                    # Create second skeleton branch
                    # (remaining branch after split)
                    # =================================================

                    arr1 = np.column_stack(coordinates2)
                    arr2 = np.column_stack(coordinates)
                    # Find rows in A that are NOT in B
                    mask = np.isin(arr1.view([('', arr1.dtype)] * arr1.shape[1]),
                                   arr2.view([('', arr2.dtype)] * arr2.shape[1]),
                                   invert=True).ravel()

                    arr3 = arr1[mask]
                    skeleton2_int = skeleton.astype(int) - skeleton1.astype(int)
                    skeleton2 = skeleton2_int != 0  # or np.array(C, dtype=bool)

                    # =================================================
                    # Clean first branch
                    # =================================================

                    pixel_graph1, coordinates1 = skeleton_to_csgraph(skeleton1)
                    branch_data1 = safe_summarize_skeleton(skeleton1)
                    branch_coords = coordinates1
                    arr = np.column_stack(branch_coords)
                    if branch_data1 is not None:
                        for ind_v2, row_v2 in branch_data1.iterrows():
                            endpoint = arr[int(row_v2["node_id_dst"]), :]
                        x, y = endpoint
                        skeleton1[x, y] = 0

                        pixel_graph1, coordinates1 = skeleton_to_csgraph(skeleton1)
                        branch_data1 = safe_summarize_skeleton(skeleton1)

                    pixel_graph2, coordinates2 = skeleton_to_csgraph(skeleton2)
                    branch_data2 = summarize(Skeleton(skeleton2), separator='_')

                    # =================================================
                    # Merge split branches back together
                    # =================================================

                    twin_mask = np.zeros(skeleton.shape, dtype=np.uint8)
                    twin_mask = twin_mask + skeleton1.astype(int) + skeleton2.astype(int)

                    skeleton = skeletonize(twin_mask > 0)
                    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
                    branch_data = summarize(Skeleton(skeleton), separator='_')

                    # Case where there are multiple twins that form junctions

            # =================================================
            # PART 2 — Resolve branch junctions
            # =================================================
            #
            # branch_type == 1 indicates branch intersections
            #
            # Goal:
            # pair connected branches according to smallest angle
            #
            # =================================================

            if (branch_data["branch_type"] == 1).any():

                ok = 1
                branch_data2 = branch_data[branch_data["branch_type"].isin([1, 2])].copy()
                branch_datas = []

                while ok == 1:

                    branch_data3 = copy.deepcopy(branch_data2)
                    matrix_angles = np.zeros((len(branch_data2), len(branch_data2))) + 180

                    for idx1 in range(len(branch_data2)):
                        row1 = branch_data2.iloc[idx1]
                        for idx2 in range(idx1 + 1, len(branch_data2)):
                            row2 = branch_data2.iloc[idx2]
                            if int(row2["branch_type"]) == 1:
                                common_pt = find_pt(row1, row2)
                                if common_pt is not None:
                                    angle = find_angle(row1, row2)
                                    matrix_angles[idx1, idx2] = angle

                    # ------------------------------------------------
                    # Select smallest angle pair
                    # These branches likely belong together
                    # ------------------------------------------------

                    min_index = np.unravel_index(np.argmin(matrix_angles), matrix_angles.shape)
                    row_pos, col_pos = min_index
                    # Map matrix positions to actual DataFrame indices
                    keep_indices = [branch_data2.index[row_pos], branch_data2.index[col_pos]]

                    # Update branch_data3 to keep only these rows
                    branch_data3 = branch_data2.loc[keep_indices].copy()

                    # Update branch_data2 to remove these rows
                    branch_data2 = branch_data2.drop(keep_indices)
                    branch_datas.append(branch_data3)

                    if len(branch_data2) == 1:
                        branch_datas.append(branch_data2)
                        ok = 0

                    if len(branch_data2) == 0:
                        ok = 0

                # =================================================
                # Extract coordinates for each branch group
                # =================================================

                for elements in branch_datas:
                    positions_f = []

                    for ind, rowi in elements.iterrows():

                        for ij, ro in branch_data.iterrows():

                            if rowi["image_coord_src_0"] == ro["image_coord_src_0"] and rowi["image_coord_src_1"] == ro[
                                "image_coord_src_1"] and rowi["image_coord_dst_0"] == ro["image_coord_dst_0"] and rowi[
                                "image_coord_dst_1"] == ro["image_coord_dst_1"]:
                                positions_f.append(Skeleton(skeleton).path_coordinates(ij))

                    final_coordinates.append(np.vstack(positions_f))

                # =================================================
                # Remove extracted coordinates from skeleton
                # =================================================

                skeleton_int = skeleton.astype(int)
                for coords in final_coordinates:
                    for r, c in coords:
                        skeleton_int[int(r), int(c)] = 0

                # =================================================
                # Remove overlapping coordinates
                # =================================================

                cleaned_coordinates = [arr.copy() for arr in final_coordinates]

                for i in range(len(cleaned_coordinates)):
                    for j in range(i + 1, len(cleaned_coordinates)):
                        arr1 = cleaned_coordinates[i]
                        arr2 = cleaned_coordinates[j]

                        # Convert to structured array for fast row comparison
                        arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
                        arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])

                        # Find common rows
                        common_mask_1 = np.isin(arr1_view, arr2_view)
                        common_mask_2 = np.isin(arr2_view, arr1_view)

                        # Remove common rows
                        cleaned_coordinates[i] = arr1[~common_mask_1.ravel()]
                        cleaned_coordinates[j] = arr2[~common_mask_2.ravel()]

                        # Step 2: Build one final binary matrix
                # =================================================
                # Rebuild final binary skeleton
                # =================================================

                final_matrix = np.zeros(skeleton.shape, dtype=np.uint8)

                for coords in cleaned_coordinates:
                    for r, c in coords:
                        final_matrix[int(r), int(c)] = 1
                        # Creation new twins
                final_matrix = skeleton_int + final_matrix
                skeleton = skeletonize(final_matrix > 0)
                pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
                branch_data = summarize(Skeleton(skeleton), separator='_')

            # =================================================
            # Check whether decomposition finished
            # =================================================

            condition1 = (branch_data["branch_distance"] > branch_data["euclidean_distance"] * 1.25).any()
            condition2 = branch_data["branch_type"].isin([1, 2]).any()

            if not (condition1 or condition2):
                for m in range(len(branch_data)):
                    final_matrix = np.zeros(skeleton.shape, dtype=np.uint8)
                    coord = Skeleton(skeleton).path_coordinates(m)
                    for r, c in coord:
                        final_matrix[int(r), int(c)] = 1

                    # plt.imshow(final_matrix)
                    # plt.show()

                THEFINALE = 1

            if condition1:
                no_change_count += 1
                if no_change_count > 10:  # or some max iteration
                    print("⚠ No branches satisfying conditions, stopping early.")
                    break

    else:
        skeleton = None
        branch_data = None

    return skeleton, branch_data

def decompose_twins_2(skeleton, l1, std):

    """
    Decompose a twin skeleton into separate branches.

    Parameters
    ----------
    skeleton : binary ndarray
        Skeletonized binary image of the twin grain.

    l1 : float
        Characteristic length parameter (passed externally).

    std : float
        Standard deviation parameter (passed externally).

    Returns
    -------
    skeleton : ndarray
        Updated skeleton after decomposition.

    branch_data : DataFrame
        Branch summary information extracted from skeleton.

    split : int
        Flag indicating whether a split operation occurred.

    Purpose
    -------
    Detects:
        1. Curved twins that likely contain merged branches
        2. Junction structures where multiple twins intersect

    Then attempts to separate them into independent branches.
    """
    # Convert skeleton into graph representation
    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)

    # Extract skeleton branch summary
    branch_data = safe_summarize_skeleton(skeleton)

    # Counter used to stop infinite decomposition loops
    no_change_count = 0

    # Controls main iterative loop
    THEFINALE = 0

    # Tracks whether a split happened
    split = 0

    # Proceed only if skeleton is valid
    if branch_data is not None:

        # Only proceed if skeleton contains one branch
        if len(branch_data) == 1:

            # Main decomposition loop
            while THEFINALE == 0:
                final_coordinates = []

                # -------------------------------------------------
                # CASE 1:
                # Twin is curved → likely consists of 2 twins
                # -------------------------------------------------

                for index, row in branch_data.iterrows():

                    # Detect curved branch
                    if row["branch_distance"] > row["euclidean_distance"]*1.25:
                        split = 1
                        ok = 0
                        skel_obj = Skeleton(skeleton)
                        branch_coords = skel_obj.path_coordinates(index)  # index of the current branch
        
                        # Create empty skeleton of the same shape
                        skeleton1 = np.zeros_like(skeleton, dtype=bool)
                        
                        # Fill only the pixels of this branch
                        for x, y in branch_coords:
                            skeleton1[int(x), int(y)] = True
                       
                        row_f = row
                        coordinates = copy.deepcopy(coordinates2)
                        ratio_d = 1.1

                        while ok == 0:
                             
                             branch_coords = coordinates
                             arr = np.column_stack(branch_coords)
                             if len(arr) == int(row_f["node_id_dst"]):
                                 endpoint = arr[int(row_f["node_id_dst"])-1,:]
                             else:
                                 endpoint = arr[int(row_f["node_id_dst"]),:]
                                 
                             x, y = endpoint
                             skeleton1[x, y] = 0 
                             new_pixel_graph, coordinates = skeleton_to_csgraph(skeleton1)
                             new_branch_data = safe_summarize_skeleton(Skeleton(skeleton1))

                             for ind, row2 in new_branch_data.iterrows():
                                 row_f = row2
                                 if row2["branch_distance"] <= row2["euclidean_distance"]*ratio_d:
                                     branch_coords = coordinates
                                     arr = np.column_stack(branch_coords)
                                     endpoint = arr[int(row_f["node_id_dst"]),:]
                                     x, y = endpoint
                                     skeleton1[x, y] = 0 
                                     true_indices = np.argwhere(skeleton1)
                                     ok = 1
                                    
                        # -------------------------------------------------
                        # Compare original and reduced branch coordinates
                        # -------------------------------------------------
                        
                        arr1 = np.column_stack(coordinates2)
                        arr2 = np.column_stack(coordinates)

                        # Find coordinates removed from original branch
                        mask = np.isin(arr1.view([('', arr1.dtype)]*arr1.shape[1]),
                                       arr2.view([('', arr2.dtype)]*arr2.shape[1]),
                                       invert=True).ravel()
                        
                        arr3 = arr1[mask]

                        # Remaining branch after subtraction
                        skeleton2_int = skeleton.astype(int) - skeleton1.astype(int)
                        skeleton2 = skeleton2_int != 0  # or np.array(C, dtype=bool)

                        # Analyze first branch
                        pixel_graph1, coordinates1 = skeleton_to_csgraph(skeleton1)
                        branch_data1 = safe_summarize_skeleton(skeleton1)
                        branch_coords = coordinates1
                        arr = np.column_stack(branch_coords)
                        if branch_data1 is not None:
                            for ind_v2, row_v2 in branch_data1.iterrows():
                                endpoint = arr[int(row_v2["node_id_dst"]),:]
                            x, y = endpoint
                            skeleton1[x, y] = 0   
                            
                            pixel_graph1, coordinates1 = skeleton_to_csgraph(skeleton1)
                            branch_data1 = safe_summarize_skeleton(skeleton1)

                        # Analyze second branch
                        pixel_graph2, coordinates2 = skeleton_to_csgraph(skeleton2)
                        branch_data2 = summarize(Skeleton(skeleton2), separator='_')

                        # Merge both branches
                        twin_mask = np.zeros(skeleton.shape, dtype=np.uint8)
                        twin_mask = twin_mask + skeleton1.astype(int) + skeleton2.astype(int)

                        # Re-skeletonize merged result
                        skeleton = skeletonize(twin_mask > 0)
                        pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
                        branch_data = summarize(Skeleton(skeleton), separator='_') 
                        
                # -------------------------------------------------
                # CASE 2:
                # Multiple twin junctions exist
                # -------------------------------------------------
                if (branch_data["branch_type"] == 1).any(): 
                    
                    ok = 1
                    branch_data2 = branch_data[branch_data["branch_type"].isin([1, 2])].copy()
                    branch_datas = []
                    
                    while ok == 1:
                        
                        branch_data3 = copy.deepcopy(branch_data2)
                        matrix_angles = np.zeros((len(branch_data2),len(branch_data2)))+180
        
                        for idx1 in range(len(branch_data2)):
                            row1 = branch_data2.iloc[idx1]
                            for idx2 in range(idx1 + 1, len(branch_data2)):
                                row2 = branch_data2.iloc[idx2]
                                if int(row2["branch_type"]) == 1:
                                    common_pt = find_pt(row1, row2)
                                    if common_pt is not None:
                                        angle = find_angle(row1, row2)
                                        matrix_angles[idx1, idx2] = angle
                                        
                                        
                        min_index = np.unravel_index(np.argmin(matrix_angles), matrix_angles.shape) 
                        row_pos, col_pos = min_index
                        # Map matrix positions to actual DataFrame indices
                        keep_indices = [branch_data2.index[row_pos], branch_data2.index[col_pos]]
                        
                        # Update branch_data3 to keep only these rows
                        branch_data3 = branch_data2.loc[keep_indices].copy()
                        
                        # Update branch_data2 to remove these rows
                        branch_data2 = branch_data2.drop(keep_indices)
                        branch_datas.append(branch_data3)
                        
                        if len(branch_data2) == 1:
                            branch_datas.append(branch_data2) 
                            ok = 0
                            
                        if len(branch_data2) == 0:
                            ok = 0
                            # Recover branch coordinates
                    for elements in branch_datas:
                        positions_f = []
                        
                        for ind, rowi in elements.iterrows():
                            
                            for ij, ro in branch_data.iterrows():
                                
                                if rowi["image_coord_src_0"] == ro["image_coord_src_0"] and rowi["image_coord_src_1"] == ro["image_coord_src_1"] and rowi["image_coord_dst_0"] == ro["image_coord_dst_0"] and rowi["image_coord_dst_1"] == ro["image_coord_dst_1"]:
                                    
                                    positions_f.append(Skeleton(skeleton).path_coordinates(ij))
                                    
                        final_coordinates.append(np.vstack(positions_f))    
                     
                    skeleton_int = skeleton.astype(int)
                    for coords in final_coordinates:
                        for r, c in coords:
                            skeleton_int[int(r), int(c)] = 0
                    
                    cleaned_coordinates = [arr.copy() for arr in final_coordinates] 
                    
                    for i in range(len(cleaned_coordinates)):
                        for j in range(i + 1, len(cleaned_coordinates)):
                            arr1 = cleaned_coordinates[i]
                            arr2 = cleaned_coordinates[j]
                    
                            # Convert to structured array for fast row comparison
                            arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
                            arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
                    
                            # Find common rows
                            common_mask_1 = np.isin(arr1_view, arr2_view)
                            common_mask_2 = np.isin(arr2_view, arr1_view)
                    
                            # Remove common rows
                            cleaned_coordinates[i] = arr1[~common_mask_1.ravel()]
                            cleaned_coordinates[j] = arr2[~common_mask_2.ravel()]
                            
                            # Step 2: Build one final binary matrix
                    final_matrix = np.zeros(skeleton.shape, dtype=np.uint8)
                    
                    for coords in cleaned_coordinates:
                        for r, c in coords:
                            final_matrix[int(r), int(c)] = 1
                            #Creation new twins
                    final_matrix = skeleton_int + final_matrix        
                    skeleton = skeletonize(final_matrix > 0)
                    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
                    branch_data = summarize(Skeleton(skeleton), separator='_')       

                # -------------------------------------------------
                # Check whether decomposition is finished
                # -------------------------------------------------
                condition1 = (branch_data["branch_distance"] > branch_data["euclidean_distance"]*1.25).any()
                condition2 = branch_data["branch_type"].isin([1, 2]).any()

                # No problematic branches remain
                if not (condition1 or condition2):
                    for m in range(len(branch_data)):
                        final_matrix = np.zeros(skeleton.shape, dtype=np.uint8)
                        coord =  Skeleton(skeleton).path_coordinates(m)
                        for r, c in coord:
                            final_matrix[int(r), int(c)] = 1
                            
                        #plt.imshow(final_matrix)
                        #plt.show()
                       
                    THEFINALE = 1

                # Safety stop to avoid infinite loop
                if condition1:
                    no_change_count += 1
                    if no_change_count > 10:  # or some max iteration
                        print("⚠ No branches satisfying conditions, stopping early.")
                        break
            
    else:
        skeleton = None
        branch_data = None    
           
    return skeleton, branch_data, split


def decompose_twins_grains(Grains, image, l1, std):
    """
    Decompose grains marked as twins into multiple branches when needed.

    Parameters
    ----------
    Grains : list
        List of Grain objects.

    image : ndarray
        Original image used to determine image dimensions.

    l1 : float
        Characteristic length parameter passed to decomposition function.

    std : float
        Standard deviation parameter passed to decomposition function.

    Returns
    -------
    Grains : list
        Updated grain list after twin decomposition.

    Purpose
    -------
    For each grain classified as a twin:

        1. Build binary mask from grain pixels
        2. Skeletonize grain mask
        3. Decompose skeleton branches
        4. If multiple branches exist:
              - Split grain into multiple grains
              - Assign pixels to nearest branch
              - Rebuild contours for each branch
        5. Update grain geometry
    """
    # Iterate through all grain objects
    for pt, grain in enumerate(Grains):

        # Create empty binary mask matching image dimensions
        twin_mask = np.zeros(np.flip(image.shape), dtype=np.uint8)

        # Process only grains marked as twins
        if grain.IsTwin:

            # Create mask from PixelList

            for (x, y) in grain.PixelList:
                try:
                    twin_mask[x, y] = 1
                except IndexError:
                    continue

            # Convert grain mask into skeleton representation
            skeleton = skeletonize(twin_mask > 0)

            # Perform twin decomposition
            skeleton, branch_data = decompose_twins(skeleton, l1, std)

            # Continue only if branch decomposition succeeded
            if branch_data is not None:

                # If multiple branches exist → split grain
                if len(branch_data) > 1:

                    num_branches = len(branch_data)
                    branch_pixel_lists = [[] for _ in range(num_branches)]

                    # Get all skeleton pixels
                    skeleton_coords = np.column_stack(np.nonzero(skeleton))

                    # Get coordinates for each branch
                    branch_coords_list = []
                    for m in range(num_branches):
                        branch_coords_list.append(Skeleton(skeleton).path_coordinates(m))

                    # Assign each skeleton pixel to the nearest branch
                    for pixel in grain.PixelList:
                        r, c = pixel
                        min_dist = np.inf
                        closest_branch = -1

                        for b_idx, coords in enumerate(branch_coords_list):
                            distances = np.sqrt((coords[:, 0] - r) ** 2 + (coords[:, 1] - c) ** 2)
                            if distances.min() < min_dist:
                                min_dist = distances.min()
                                closest_branch = b_idx

                        if closest_branch >= 0:
                            branch_pixel_lists[closest_branch].append((r, c))

                    # --------------------------------------------
                    # Build new grain object for each branch
                    # --------------------------------------------

                    for branch_idx, branch_pixels in enumerate(branch_pixel_lists):
                        if not branch_pixels:
                            continue  # Skip empty branches

                        branch_mask = np.zeros(skeleton.shape, dtype=np.uint8)

                        for r, c in branch_pixels:
                            branch_mask[int(r), int(c)] = 1

                        bool_ske = branch_mask.astype(bool)

                        labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
                        contours = []
                        contours = find_contours(labeled_array_ske, 0.5)
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
                        contour2 = np.array(contour, np.int32)
                        branch_pixels = np.array(branch_pixels)

                        # ----------------------------------------
                        # Replace original grain with first branch
                        # ----------------------------------------

                        if branch_idx == 0:
                            contour_array = np.array(contour, dtype=np.int32)
                            center_x = np.mean(branch_pixels[:, 0])
                            center_y = np.mean(branch_pixels[:, 1])
                            gr_before = Grains[pt]
                            ID = gr_before.ID
                            ID2 = gr_before.ID2

                            gr = Grain(branch_pixels, contour_array, (center_x, center_y), len(branch_pixels[:, 0]), 1,
                                       ID, is_twin=True)
                            gr.DilatedContourPoints = np.flip(contour_array, axis=1)
                            gr.SkeletonCoord = branch_coords_list[branch_idx]
                            Grains[pt] = gr

                        # ----------------------------------------
                        # Create new grain for additional branches
                        # ----------------------------------------

                        else:
                            contour_array = np.array(contour, dtype=np.int32)
                            center_x = np.mean(branch_pixels[:, 0])
                            center_y = np.mean(branch_pixels[:, 1])
                            gr = Grain(branch_pixels, contour_array, (center_x, center_y), len(branch_pixels[:, 0]), 1,
                                       len(Grains) + branch_idx, is_twin=True)
                            gr.DilatedContourPoints = np.flip(contour_array, axis=1)
                            gr.SkeletonCoord = branch_coords_list[branch_idx]
                            Grains.append(gr)

                # --------------------------------------------
                # Single branch case
                # Store skeleton coordinates only
                # --------------------------------------------

                else:

                    grain.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)
            # --------------------------------------------
            # Decomposition failed
            # Use original skeleton instead
            # --------------------------------------------

            else:
                skeleton = skeletonize(twin_mask > 0)
                true_indices = np.where(skeleton == True)
                if len(true_indices[0]) > 1:
                    grain.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)
                elif (len(grain.SkeletonCoord) == 0) and len(true_indices[0]) == 1:
                    grain.SkeletonCoord = true_indices

    return Grains
    
def decompose_twins_grains_2(Grains, image, l1, std):

    """
    Decompose twin grains into individual branches based on their skeleton structure.

    Parameters
    ----------
    Grains : list of Grain
        List of Grain objects, some of which are flagged as twins (IsTwin).
    image : ndarray
        Reference image used only for its shape, to size the twin mask.
    l1 : float
        Parameter passed through to decompose_twins_2 controlling branch
        decomposition sensitivity.
    std : float
        Standard deviation parameter passed through to decompose_twins_2.

    Returns
    -------
    Grains : list of Grain
        Updated list of grains. Twin grains whose skeleton splits into
        multiple branches are replaced/appended as separate Grain objects,
        one per branch. Non-splitting twins have their SkeletonCoord updated
        in place.

    Purpose
    -------
    For every grain flagged as a twin, builds a binary mask of its pixels,
    skeletonizes it, and checks whether the skeleton decomposes into
    multiple branches (a "split"). If so, each branch is reassigned its own
    pixels (by nearest-distance), contour, and Grain object, so that a single
    twin region is split into multiple twin grains. If the skeleton doesn't
    split, the grain's SkeletonCoord is simply updated to reflect its
    skeleton path.
    """

    for pt, grain in enumerate(Grains):

        # Mask sized to match the image dimensions (flipped to match indexing convention)
        twin_mask = np.zeros(np.flip(image.shape), dtype=np.uint8)
        
        if grain.IsTwin:

                
            # Create mask from PixelList
        
            for (x, y) in grain.PixelList:
                try:
                    twin_mask[x, y] = 1
                except IndexError:
                    continue

            # Reduce the twin region to its 1-pixel-wide skeleton
            skeleton = skeletonize(twin_mask > 0)

            # Attempt to decompose the skeleton into separate branches
            skeleton, branch_data, split = decompose_twins_2(skeleton, l1, std)
            if branch_data is not None:
                if len(branch_data) > 1 and split == 1:
                    # Skeleton has multiple branches and should be split into
                    # separate grains

                    num_branches = len(branch_data)
                    branch_pixel_lists = [[] for _ in range(num_branches)]
                    
                    # Get all skeleton pixels
                    skeleton_coords = np.column_stack(np.nonzero(skeleton))
                    
                    # Get coordinates for each branch
                    branch_coords_list = []
                    for m in range(num_branches):
                        branch_coords_list.append(Skeleton(skeleton).path_coordinates(m))

                    # Assign each twin pixel to its nearest branch by
                    # Euclidean distance to that branch's skeleton coordinates
                    for pixel in grain.PixelList:
                        r, c = pixel
                        min_dist = np.inf
                        closest_branch = -1
                        
                        for b_idx, coords in enumerate(branch_coords_list):
                            distances = np.sqrt((coords[:, 0] - r)**2 + (coords[:, 1] - c)**2)
                            if distances.min() < min_dist:
                                min_dist = distances.min()
                                closest_branch = b_idx
                        
                        if closest_branch >= 0:
                            branch_pixel_lists[closest_branch].append((r, c))
                    
                    # Build a new Grain object for each branch, deriving its
                    # contour from the pixels assigned to that branch
                    
                    for branch_idx, branch_pixels in enumerate(branch_pixel_lists):
                        if not branch_pixels:
                            continue  # Skip empty branches
                            
                        branch_mask = np.zeros(skeleton.shape, dtype=np.uint8)
                        
                        for r, c in branch_pixels:
                            branch_mask[int(r), int(c)] = 1
                        
                        bool_ske = branch_mask.astype(bool)
                        
                        labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
                        contours = []
                        contours = find_contours(labeled_array_ske, 0.5)
                        contour_points_list = []
                        for contour in contours:
                            contour = np.round(contour).astype(int)
                            contour_points = []
                    
                            for point in contour:
                                y, x = point
                                corrected_x = max(0, math.floor(x)-1) 
                                corrected_y = max(0, math.floor(y)-1) 
                                contour_points.append((corrected_x, corrected_y))
                                
                                #contour_image[math.floor(y)-1, math.floor(x)-1] = 255
                            contour_points_list.append(contour_points)      
                        contour2 = np.array(contour, np.int32)
                        branch_pixels = np.array(branch_pixels)
    
                        if branch_idx == 0:
                            # First branch reuses the original grain's ID and
                            # replaces it in place
                            contour_array = np.array(contour, dtype=np.int32)
                            center_x = np.mean(branch_pixels[:,0])
                            center_y = np.mean(branch_pixels[:,1])
                            gr_before = Grains[pt]
                            ID = gr_before.ID
                            ID2 = gr_before.ID2
                            
                            gr = Grain(branch_pixels,contour_array,(center_x,center_y),len(branch_pixels[:,0]),1,ID, is_twin=True)
                            gr.DilatedContourPoints = np.flip(contour_array, axis = 1)
                            gr.SkeletonCoord = branch_coords_list[branch_idx]
                            Grains[pt] = gr
                            
                        else:
                            contour_array = np.array(contour, dtype=np.int32)
                            center_x = np.mean(branch_pixels[:,0])
                            center_y = np.mean(branch_pixels[:,1])
                            gr = Grain(branch_pixels,contour_array,(center_x,center_y),len(branch_pixels[:,0]),1,len(Grains)+branch_idx, is_twin=True)
                            gr.DilatedContourPoints = np.flip(contour_array, axis = 1)
                            gr.SkeletonCoord = branch_coords_list[branch_idx]
                            Grains.append(gr)
                            
                else:
                    # Skeleton has branch data but doesn't need splitting;
                    # just record the combined skeleton coordinates on the
                    # existing grain
                    num_branches = len(branch_data)
                    branch_pixel_lists = [[] for _ in range(num_branches)]
                    
                    # Get all skeleton pixels
                    skeleton_coords = np.column_stack(np.nonzero(skeleton))
                    
                    # Get coordinates for each branch
                    branch_coords_list = []
                    for m in range(num_branches):
                        branch_coords_list.append(Skeleton(skeleton).path_coordinates(m))
                    all_branch_coords = np.concatenate(branch_coords_list, axis=0)
                    grain.SkeletonCoord = all_branch_coords                    
            else :
                # No branch data returned; fall back to a single-path skeleton
                skeleton = skeletonize(twin_mask > 0)
                true_indices = np.where(skeleton == True)
                if len(true_indices[0]) > 1:
                    grain.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)    
                elif (len(grain.SkeletonCoord) == 0) and len(true_indices[0]) == 1:
                    grain.SkeletonCoord = true_indices
                
    return Grains

def find_overlapping_grains(grains):
    """
    Identify pairs of grains that share one or more pixels.

    Parameters
    ----------
    grains : list of Grain
        List of Grain objects, each with a PixelList attribute.

    Returns
    -------
    overlapping_grains : list of tuple of Grain
        List of (grain_a, grain_b) pairs whose pixel lists overlap.
    ID_grains : list of tuple of (int, int, int)
        List of (index_a, index_b, overlap_count) describing the same pairs
        by their position in the input list, along with how many pixels
        they share.

    Purpose
    -------
    Builds a pixel-to-grain-indices lookup, then scans for pixels claimed by
    more than one grain to detect overlapping segmentation results. Used as
    a precursor step to resolving overlaps (see remove_overlapping_pixels).
    """

    pixel_to_grains = defaultdict(list)

    # Map every pixel to the IDs of grains touching it
    for idx, grain in enumerate(grains):
        for pixel in grain.PixelList:
            pixel_to_grains[tuple(pixel)].append(idx)

    # Find pixels where more than one grain exists
    overlaps = defaultdict(int)
    for pixel, grain_ids in pixel_to_grains.items():
        if len(grain_ids) > 1:
            # For every pair of grains at this pixel, increment overlap count
            from itertools import combinations
            for i, j in combinations(sorted(grain_ids), 2):
                overlaps[(i, j)] += 1

    # Format output to match your original structure
    ID_grains = [(i, j, count) for (i, j), count in overlaps.items()]
    overlapping_grains = [(grains[i], grains[j]) for (i, j, count) in ID_grains]

    return overlapping_grains, ID_grains

def remove_overlapping_pixels_ATRISK(grain1, grain2, size):
    """
    Resolve pixel overlap between two grains, preferring the one flagged AtRisk.

    Parameters
    ----------
    grain1 : Grain
        First grain in the overlapping pair.
    grain2 : Grain
        Second grain in the overlapping pair.
    size : tuple of int
        (width, height) of the image, used to bound and rasterize pixels.

    Returns
    -------
    lower_conf_grain : Grain
        The grain flagged AtRisk, with overlapping pixels removed and its
        PixelList, ContourPoints, and size updated accordingly. Returns
        unchanged if neither grain is flagged AtRisk.

    Purpose
    -------
    Legacy/alternate overlap-resolution routine that strips shared pixels
    from whichever grain is marked AtRisk (rather than by confidence score,
    as in remove_overlapping_pixels), then recomputes that grain's contour
    from the remaining pixels via erosion and connected-component analysis.
    """
    width, height = size

    # Determine which grain has the lowest confidence
    if grain1.AtRisk:
        lower_conf_grain, higher_conf_grain = grain1, grain2
    elif grain2.AtRisk:
        lower_conf_grain, higher_conf_grain = grain2, grain1
    else:
        return lower_conf_grain  # No action if neither grain is at risk

    # Convert PixelLists to sets of tuples for easier manipulation
    lower_pixels_set = set(map(tuple, lower_conf_grain.PixelList))
    higher_pixels_set = set(map(tuple, higher_conf_grain.PixelList))

    # Find overlapping pixels
    overlap_pixels = lower_pixels_set.intersection(higher_pixels_set)

    # Remove overlapping pixels from the grain with lower confidence
    updated_pixels_set = lower_pixels_set - overlap_pixels

    # Convert the updated set back to a numpy array
    updated_pixels_list = np.array(list(updated_pixels_set), dtype=np.int32)

    # Update the contour of the lower confidence grain
    if len(updated_pixels_list) > 0:
        # Create a blank mask
        mask = np.zeros((width, height), dtype=np.uint8)

        # Ensure pixel coordinates are within the bounds
        valid_pixels = [(x, y) for (x, y) in updated_pixels_list if 0 <= x < width and 0 <= y < height]

        # Set remaining pixels to white on the mask
        for (x, y) in valid_pixels:
            mask[x, y] = 255

        # Erode the mask
        kernel = np.ones((1, 1), np.uint8)  # Define a kernel for erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        # Find contours from the eroded mask
        bool_ske = eroded_mask.astype(bool)
        labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)

        if num_features_ske > 1:
            # If there are multiple features, keep the largest one
            regions = regionprops(labeled_array_ske)
            largest_region = max(regions, key=lambda r: r.area)

            # Remove the pixels of the other regions
            for region in regions:
                if region != largest_region:
                    for coord in region.coords:
                        eroded_mask[coord[0], coord[1]] = 0

        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            new_contour = contours[0].reshape(-1, 2)  # Convert from (n,1,2) to (n,2)
        else:
            new_contour = np.array([], dtype=np.int32)

        # Update the pixel list from the eroded mask
        updated_pixels_list = np.argwhere(eroded_mask == 255)
    else:
        # If no pixels are left, the grain is removed
        new_contour = np.array([], dtype=np.int32)
        updated_pixels_list = np.array([], dtype=np.int32)
    # Invert x and y for the contour and pixel list

    # Update the grain with the new contour and size
    lower_conf_grain.PixelList = updated_pixels_list
    lower_conf_grain.ContourPoints = new_contour
    lower_conf_grain.size = len(updated_pixels_list)

    return lower_conf_grain

def remove_overlapping_pixels(grain1, grain2, size):
    """
    Resolve pixel overlap between two grains, preferring the higher-confidence one.

    Parameters
    ----------
    grain1 : Grain
        First grain in the overlapping pair, must have a confidence attribute.
    grain2 : Grain
        Second grain in the overlapping pair, must have a confidence attribute.
    size : tuple of int
        (width, height) of the image, used to bound and rasterize pixels.

    Returns
    -------
    lower_conf_grain : Grain
        The lower-confidence grain (or, on a confidence tie, the larger one),
        with overlapping pixels removed and its PixelList, ContourPoints,
        and size updated accordingly.

    Purpose
    -------
    Main overlap-resolution routine used by handle_overlapping_grains.
    Decides which of the two grains should lose the shared pixels (lower
    model confidence wins precedence; ties are broken by size), strips the
    overlap from that grain, and recomputes its contour from the remaining
    pixels via connected-component analysis, keeping only the smallest
    region if the remaining pixels split into multiple disconnected blobs.
    """
    width, height = size
    # Determine which grain has the lowest confidence
    if grain1.confidence < grain2.confidence:
        lower_conf_grain, higher_conf_grain = grain1, grain2
    elif grain1.confidence == grain2.confidence:
        if grain1.size > grain2.size:
            lower_conf_grain, higher_conf_grain = grain2, grain1
        else:
            higher_conf_grain, lower_conf_grain = grain2, grain1
    else:
        lower_conf_grain, higher_conf_grain = grain2, grain1
    
    # Convert PixelLists to sets of tuples for easier manipulation
    lower_pixels_set = set(map(tuple, lower_conf_grain.PixelList))
    higher_pixels_set = set(map(tuple, higher_conf_grain.PixelList))
    
    # Find overlapping pixels
    overlap_pixels = lower_pixels_set.intersection(higher_pixels_set)
    
    # Remove overlapping pixels from the grain with lower confidence
    updated_pixels_set = lower_pixels_set - overlap_pixels
    
    # Convert the updated set back to a numpy array
    updated_pixels_list = np.array(list(updated_pixels_set), dtype=np.int32)
    # Update the contour of the lower confidence grain
    if len(updated_pixels_list) > 0:
        # Create a blank mask
        mask = np.zeros((width, height), dtype=np.uint8)
        # Ensure pixel coordinates are within the bounds
        valid_pixels = [(x, y) for (x, y) in updated_pixels_list if 0 <= x < width and 0 <= y < height]
        
        # Set remaining pixels to white on the mask
        for (x, y) in valid_pixels:
            mask[x, y] = 255
               # Find contours from the mask
                  # Find contours from the mas
        bool_ske = mask.astype(bool)
        labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
        if num_features_ske > 1:
    # If there are multiple features, labeled_array_ske the smallest one
            regions = regionprops(labeled_array_ske)
            smallest_region = min(regions, key=lambda r: r.area)
    
    # Remove the pixels of the smallest region
            for coord in smallest_region.coords:
                mask[coord[0], coord[1]] = 0
             
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            new_contour = contours[0].reshape(-1, 2)  # Convert from (n,1,2) to (n,2)
        else:
            new_contour = np.array([], dtype=np.int32)
    else:
        # If no pixels are left, the grain is removed
        new_contour = np.array([], dtype=np.int32)
    
    # Update the grain with the new contour and size
    lower_conf_grain.PixelList = updated_pixels_list
    lower_conf_grain.ContourPoints = new_contour
    lower_conf_grain.size = len(updated_pixels_list)
    # Create a blank mask
    mask = np.zeros((width, height), dtype=np.uint8)
    
    # Set the pixels of the updated grain to white on the mask
    for (x, y) in lower_conf_grain.PixelList:
        if 0 <= x < width and 0 <= y < height:
            mask[x, y] = 255
        
    return lower_conf_grain

def handle_overlapping_grains(overlapping_grains,size):
    """
    Resolve pixel overlap for every pair of overlapping grains.

    Parameters
    ----------
    overlapping_grains : list of tuple of Grain
        List of (grain_a, grain_b) pairs as returned by find_overlapping_grains.
    size : tuple of int
        (width, height) of the image, passed through to remove_overlapping_pixels.

    Returns
    -------
    None
        Grains are updated in place via remove_overlapping_pixels.

    Purpose
    -------
    Batch driver that applies remove_overlapping_pixels to every overlapping
    pair found by find_overlapping_grains, so that the full set of grains
    ends up with no remaining pixel overlaps.
    """

    for grain1, grain2 in overlapping_grains:
        remove_overlapping_pixels(grain1, grain2, size)
        
def image_size(image_path):
    """
    Load an image and return it in RGB along with its (width, height).

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.

    Returns
    -------
    image_rgb : ndarray
        Image converted from BGR (OpenCV's default) to RGB.
    size : tuple of int
        (width, height) of the image.

    Purpose
    -------
    Convenience loader used wherever both the RGB image and its pixel
    dimensions are needed together, e.g. for building masks or normalising
    contour coordinates.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2] 
    return image_rgb,(width, height)

def read_contours(txt_file, size, mojo):
    """
    Parse normalised contour coordinates and confidences from a YOLO-style label file.

    Parameters
    ----------
    txt_file : str
        Path to the text file containing one annotation per line, with
        normalised (0-1) polygon coordinates.
    size : tuple of int
        (width, height) used to scale normalised coordinates back to pixels.
    mojo : int
        Format selector. 0: line ends with a trailing confidence value to be
        parsed separately. 1: no trailing confidence value; confidence is
        defaulted to 1 for every contour.

    Returns
    -------
    contours : list of ndarray
        List of (N, 2) arrays of (x, y) pixel coordinates, one array per
        annotation line.
    confidences : list
        List of confidence values (floats as strings if mojo == 0, or
        literal 1 if mojo == 1), one per contour.

    Purpose
    -------
    Reads YOLO-segmentation-style label files where each line encodes a
    class label followed by a flattened, normalised polygon (and optionally
    a trailing confidence score), and converts them into pixel-space contour
    arrays usable with OpenCV/skimage.
    """
    contours = []
    confidences =[]
    with open(txt_file, 'r') as file:
        for line in file:
            if line.strip():
                if mojo == 0:
                    parts = line.strip().split()
                    points = parts[1:-1]  # Skip the first element as it is just a label
                    
                    confidences.append(parts[len(parts)-1])
                if mojo == 1:
                    parts = line.strip().split()
                    points = parts[1:]  # Skip the first element as it is just a label
                    
                    confidences.append(1)

                if len(points) > 0:
                    contour = [(float(points[i]) * (size[0]), float(points[i + 1]) * (size[1])) for i in range(0, len(points), 2)]
                    contours.append(np.array(contour))

    return contours, confidences
