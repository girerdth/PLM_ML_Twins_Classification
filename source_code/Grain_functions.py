# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:33:50 2024

@author: ezxtg6
"""
from ultralytics import YOLO
import cv2
import os
import numpy as np
from skimage.measure import regionprops, label
import copy
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label, find_contours
import math

def get_integer_points_inside_contour(contour):
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

def extract_grains_from_contour_mask(contour_mask, image_size, starting_id=1000):
    labeled_mask, _ = label(~contour_mask.astype(bool), return_num=True)
    grains = []
    from Grain_functions import Grain  # Avoid circular import

    for i, region in enumerate(regionprops(labeled_mask)):
        if region.area > 0:
            coords = region.coords
            dilated_mask = np.zeros(image_size, dtype=np.uint8)
            for (x, y) in coords:
                try:
                    dilated_mask[x, y] = 1
                except IndexError:
                    continue

            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(dilated_mask, kernel, iterations=1)
            dilated_coords = np.column_stack(np.where(dilated == 1))

            contours, _ = cv2.findContours(dilated.T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_points = contours[0].reshape(-1, 2) if contours else np.array([])

            grain = Grain(
                pixel_list=dilated_coords,
                contour_points=contour_points,
                centroid=(np.mean(coords[:, 0]), np.mean(coords[:, 1])),
                size=len(coords),
                confidence=0.5,
                ID=starting_id + i
            )
            grains.append(grain)

    return grains

def cleaning_grains(Grains,size):
    Grains = sorted(Grains, key=lambda x: x.size, reverse=True) 
    Grains2 = copy.deepcopy(Grains)
    overlapping_grains, ID_grains = find_overlapping_grains(Grains2)
    handle_overlapping_grains(overlapping_grains,size)
    grain_matrix = np.zeros(size, dtype=np.int32)
    mask = np.zeros(size, dtype=np.uint8)
    for grain in Grains2:
        for (x, y) in grain.PixelList:
            try:
                grain_matrix[x, y] = grain.ID
                mask[x, y] = 1
            except IndexError as e:
               # print(f"IndexError: Grain ID {grain.ID} caused an out-of-bounds error at position ({x}, {y})")
                continue 
    sorted_grains = sorted(Grains2, key=lambda g: g.confidence, reverse=True)
    grain_matrix2 = copy.deepcopy(grain_matrix)    

    for grain in sorted_grains:
        # Create an empty mask for the grain contour
        mask2 = np.zeros(size, dtype=np.uint8)
        
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
    max_existing_id = max(grain.ID for grain in Grains2)
    n = 1
    for region in regionprops(labeled_zeros):
        if region.area > 0:  # Ignore very small regions if needed
         # Extract the coordinates of the current region
            coords = region.coords
            new_id = max_existing_id + n + 1
         
            new_mask = np.zeros(size, dtype=np.uint8)
            for (x, y) in coords:
                try:
                    new_mask[x, y] = 1
                except IndexError as e:
         #print("toto")
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
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Create a contour mask where the gradient is positive
    contour_mask = (gradient_magnitude > 0).astype(np.uint8)
    contour_mask_bool = contour_mask.astype(bool)

    skeletonized_contour_mask = skeletonize(contour_mask_bool).astype(np.uint8)
    # Dilate the contour mask to enhance contours
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated_contours = cv2.dilate(skeletonized_contour_mask, kernel_dilate, iterations=1)
    def extract_contours_from_mask(mask):
        grains = []
        labeled_zeros, num_labels = label(~mask.astype(bool), return_num=True)
        i = 1
        
        for region in regionprops(labeled_zeros):
            if region.area > 0:  # Ignore very small regions if needed
                # Extract the coordinates of the current region
                
                coords = region.coords
          
                coords_vf = coords[:, [1, 0]]
                new_id = max_existing_id + i + 1
                # Create a mask for the new grain and find its contour
                new_mask = np.zeros(size, dtype=np.uint8)
                for (x, y) in coords:
                    try:
                        new_mask[x, y] = 1
                    except IndexError as e:
                        print("toto")
                        continue  # Skip this pixel and continue with the next one
                kernel_dilate = np.ones((3, 3), np.uint8)
                dilated_mask  = cv2.dilate(new_mask, kernel_dilate, iterations=1)
                dilated_coords = np.column_stack(np.where(dilated_mask == 1))    
            # Create a new grain
                new_grain = Grain(
                    pixel_list=dilated_coords,
                    contour_points=np.array([]),  # Placeholder; will be updated later
                    centroid=(np.mean(coords[:, 0]), np.mean(coords[:, 1])),
                    size=len(coords),
                    confidence=0.5,  # Default confidence for new grains
                    ID=new_id
                )
                
                contours, _ = cv2.findContours(np.transpose(dilated_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    new_grain.ContourPoints = contours[0].reshape(-1, 2)  # Update contour
                # Update the grain_matrix with the new grain ID
                grains.append(new_grain)

                i = i + 1
        return grains

    # Extract contours from the smoothed mask
    Seg_Grain_Final = extract_contours_from_mask(dilated_contours)
    rotated_contour = np.rot90(dilated_contours)
    # Take the symmetry (flip horizontally)
    symmetric_contour = np.flipud(rotated_contour)
    final_contour = cv2.resize(symmetric_contour, size,interpolation= cv2.INTER_NEAREST)
    
    return Seg_Grain_Final, final_contour

def get_integer_points_inside_contour_gt(contour):
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


def merge_grain(grain1, grain2, new_id=None):
    """
    Merge two Grain objects into a single new Grain, with contour recalculated.
    """
    # --- Combine pixel lists ---
    pixel_list = np.vstack([grain1.PixelList, grain2.PixelList])
    
    # --- Compute new centroid ---
    centroid = np.mean(pixel_list, axis=0)
    
    # --- Recompute contour using OpenCV ---
    # Convert to integer pixel coordinates
    pixels_int = np.round(pixel_list).astype(np.int32)
    
    # Make a binary mask just large enough to contain all pixels
    max_yx = np.max(pixels_int, axis=0).astype(int) + 2
    mask = np.zeros((max_yx[0]+1, max_yx[1]+1), np.uint8)
    mask[pixels_int[:,0], pixels_int[:,1]] = 255

    # Find contour (OpenCV expects (x, y) order)
    contours, _ = cv2.findContours(mask.T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour_points = np.squeeze(contours[0])
    else:
        contour_points = np.zeros((0, 2))

    # --- Combine other numeric attributes ---
    size = grain1.size + grain2.size
    confidence = np.mean([grain1.confidence, grain2.confidence])
    merged_id = new_id if new_id is not None else grain1.ID

    # --- Create the merged grain ---
    merged_grain = Grain(
        pixel_list=pixel_list,
        contour_points=contour_points,
        centroid=centroid,
        size=size,
        confidence=confidence,
        ID=merged_id
    )
    
    
    merged_grain.DilatedContourPoints = np.array([(y, x) for x, y in contour_points])
    # --- Combine logical/attribute information ---
    merged_grain.IsTwin = grain1.IsTwin or grain2.IsTwin
    
    # Average or preserve orientation values if both exist
    if (grain1.Azimuth is not None) and (grain2.Azimuth is not None):
        merged_grain.Azimuth = (grain1.Azimuth*grain1.size + grain2.Azimuth*grain2.size)/size
    else:
        merged_grain.Azimuth = grain1.Azimuth or grain2.Azimuth

    if (grain1.Inclination is not None) and (grain2.Inclination is not None):
        merged_grain.Inclination = (grain1.Inclination*grain1.size + grain2.Inclination*grain2.size)/size
    else:
        merged_grain.Inclination = grain1.Inclination or grain2.Inclination
    
    # --- Merge neighbour/friend information (remove duplicates) ---
    merged_grain.Neighbours = list(set(grain1.Neighbours + grain2.Neighbours))
    merged_grain.Friends = list(set(grain1.Friends + grain2.Friends))
    
    # --- Merge grayscale info (optional averaging) ---
    merged_grain.GrayMean = grain1.GrayMean
    merged_grain.GrayCount = grain1.GrayCount
    
    # --- Update derived geometric properties ---
    merged_grain.update_ellipsoid()
    
    return merged_grain

def get_integer_points_inside_contour(contour):
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

class Grain:
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
            scale: 'sqrt2' -> semi-axes = sqrt(2*lambda)
                   'sigma1' -> semi-axes = sqrt(lambda)
            returns: (a, b, tau) with a >= b
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
        scale: 'sqrt2' -> semi-axes = sqrt(2*lambda)
               'sigma1' -> semi-axes = sqrt(lambda)
        returns: (a, b, tau) with a >= b
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

    def add_twinning_area2(self, area):
        self.twinning_area2 += area
        return self.twinning_area2

    def add_twinning_area3(self, area):
        self.twinning_area3 += area
        return self.twinning_area3

    def is_twinning(self, is_twin):
        self.IsTwin = is_twin
        return self.IsTwin

    def set_orientation(self, azimuth, inclination):
        self.Azimuth = azimuth
        self.Inclination = inclination

    def set_gray_analysis(self, gray_mean_values, position, threshold=10):
        """
        Sets grayscale profile and count of images where gray mean is below threshold.
        """
        self.GrayMean = gray_mean_values
        self.Position = position
        self.GrayCount = sum(val < threshold for val in gray_mean_values)
    
    def set_neighbours(self,neigh):
        self.Neighbours = neigh
    def add_friends(self,friends):
        
        self.HaveFriends = True
        toto = self.Friends
        toto.append(friends)
        self.Friends = toto  
        
from skimage.morphology import skeletonize, thin
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize
  
def find_pt(row1, row2):
        
    pt11 = np.array([int(row1["image_coord_src_0"]), int(row1["image_coord_src_1"])])
    pt21 = np.array([int(row1["image_coord_dst_0"]), int(row1["image_coord_dst_1"])])
         
    pt12 = np.array([int(row2["image_coord_src_0"]), int(row2["image_coord_src_1"])])
    pt22 = np.array([int(row2["image_coord_dst_0"]), int(row2["image_coord_dst_1"])])
    
    if np.array_equal(pt11, pt12) or np.array_equal(pt11, pt22):
        return pt11
    if np.array_equal(pt21, pt12) or np.array_equal(pt21, pt22):
        return pt21
    
    return None
  

def measure_ellipsoid(Grains):
    
    for gr in Grains:
        gr = Grain()
    
    
    return Grains
def find_angle(row1, row2):
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

def safe_summarize_skeleton(skeleton_img):
    # Check if skeleton has any nonzero pixels
    if not np.any(skeleton_img):
        print("⚠ Skeleton is empty, skipping.")
        return None

    try:
        return summarize(Skeleton(skeleton_img), separator='_')
    except ValueError as e:
        print(f"⚠ Error processing skeleton: {e}")
        return None
    
def find_grain_by_ID(grains, ID):
    
    for gr in grains:
        if gr.ID == ID:
            return gr
        
    return None
    

def find_grain_by_ID_index(grains, ID):
    
    for i, gr in enumerate(grains):
        if gr.ID == ID:
            return i
        
    return None

def decompose_twins(skeleton, l1, std):
    
    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
    branch_data = safe_summarize_skeleton(skeleton)
    no_change_count = 0
    THEFINALE = 0

    if branch_data is not None:
        while THEFINALE == 0:
            final_coordinates = []
            # Case where twin is not flat because it's 2 twins
            for index, row in branch_data.iterrows():
                if row["branch_distance"] > row["euclidean_distance"]*1.25:

                    ok = 0
                    skel_obj = Skeleton(skeleton)
                    branch_coords = skel_obj.path_coordinates(index)  # index of the current branch
    
                    # Create empty skeleton of the same shape
                    skeleton1 = np.zeros_like(skeleton, dtype=bool)
                    
                    # Fill only the pixels of this branch
                    for x, y in branch_coords:
                        skeleton1[int(x), int(y)] = True
                   
                    row_f = row
                    poto = 0
                    coordinates = copy.deepcopy(coordinates2)
                    ratio_d = 1.1
                    tututu = 0
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
                         new_branch_data = summarize(Skeleton(skeleton1), separator='_')
                         tututu = tututu + 1
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
                                
                                     
                    
                    arr1 = np.column_stack(coordinates2)
                    arr2 = np.column_stack(coordinates)
                    # Find rows in A that are NOT in B
                    mask = np.isin(arr1.view([('', arr1.dtype)]*arr1.shape[1]),
                                   arr2.view([('', arr2.dtype)]*arr2.shape[1]),
                                   invert=True).ravel()
                    
                    arr3 = arr1[mask]
                    skeleton2_int = skeleton.astype(int) - skeleton1.astype(int)
                    skeleton2 = skeleton2_int != 0  # or np.array(C, dtype=bool)
                    
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
                    
                    pixel_graph2, coordinates2 = skeleton_to_csgraph(skeleton2)
                    branch_data2 = summarize(Skeleton(skeleton2), separator='_') 
                    
                    twin_mask = np.zeros(skeleton.shape, dtype=np.uint8)
                    twin_mask = twin_mask + skeleton1.astype(int) + skeleton2.astype(int)
                    
                    skeleton = skeletonize(twin_mask > 0)
                    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
                    branch_data = summarize(Skeleton(skeleton), separator='_') 
                    
            #Case where there are multiple twins that form junctions
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
                    
            condition1 = (branch_data["branch_distance"] > branch_data["euclidean_distance"]*1.25).any()
            condition2 = branch_data["branch_type"].isin([1, 2]).any()
                    
            if not (condition1 or condition2):
                for m in range(len(branch_data)):
                    print(m)
                    final_matrix = np.zeros(skeleton.shape, dtype=np.uint8)
                    coord =  Skeleton(skeleton).path_coordinates(m)
                    for r, c in coord:
                        final_matrix[int(r), int(c)] = 1
                        
                    #plt.imshow(final_matrix)
                    #plt.show()
                   
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
    
    pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
    branch_data = safe_summarize_skeleton(skeleton)
    no_change_count = 0
    THEFINALE = 0
    split = 0

    if branch_data is not None:
        if len(branch_data) == 1:
            while THEFINALE == 0:
                final_coordinates = []
                # Case where twin is not flat because it's 2 twins
                for index, row in branch_data.iterrows():
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
                        poto = 0
                        coordinates = copy.deepcopy(coordinates2)
                        ratio_d = 1.1
                        tututu = 0
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
                             new_branch_data = summarize(Skeleton(skeleton1), separator='_')
                             tututu = tututu + 1
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
                                    
                                         
                        
                        arr1 = np.column_stack(coordinates2)
                        arr2 = np.column_stack(coordinates)
                        # Find rows in A that are NOT in B
                        mask = np.isin(arr1.view([('', arr1.dtype)]*arr1.shape[1]),
                                       arr2.view([('', arr2.dtype)]*arr2.shape[1]),
                                       invert=True).ravel()
                        
                        arr3 = arr1[mask]
                        skeleton2_int = skeleton.astype(int) - skeleton1.astype(int)
                        skeleton2 = skeleton2_int != 0  # or np.array(C, dtype=bool)
                        
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
                        
                        pixel_graph2, coordinates2 = skeleton_to_csgraph(skeleton2)
                        branch_data2 = summarize(Skeleton(skeleton2), separator='_') 
                        
                        twin_mask = np.zeros(skeleton.shape, dtype=np.uint8)
                        twin_mask = twin_mask + skeleton1.astype(int) + skeleton2.astype(int)
                        
                        skeleton = skeletonize(twin_mask > 0)
                        pixel_graph, coordinates2 = skeleton_to_csgraph(skeleton)
                        branch_data = summarize(Skeleton(skeleton), separator='_') 
                        
                #Case where there are multiple twins that form junctions
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
                        
                condition1 = (branch_data["branch_distance"] > branch_data["euclidean_distance"]*1.25).any()
                condition2 = branch_data["branch_type"].isin([1, 2]).any()
                        
                if not (condition1 or condition2):
                    for m in range(len(branch_data)):
                        print(m)
                        final_matrix = np.zeros(skeleton.shape, dtype=np.uint8)
                        coord =  Skeleton(skeleton).path_coordinates(m)
                        for r, c in coord:
                            final_matrix[int(r), int(c)] = 1
                            
                        #plt.imshow(final_matrix)
                        #plt.show()
                       
                    THEFINALE = 1
            
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
    
    for pt, grain in enumerate(Grains):
    
        twin_mask = np.zeros(np.flip(image.shape), dtype=np.uint8)
        
        if grain.IsTwin:
            if grain.ID == 13:
                pttt = 1
                
            # Create mask from PixelList
        
            for (x, y) in grain.PixelList:
                try:
                    twin_mask[x, y] = 1
                except IndexError:
                    continue
            if grain.ID == 129:
                titit = 1
                
           
            skeleton = skeletonize(twin_mask > 0)
            
            skeleton, branch_data = decompose_twins(skeleton, l1, std)
            if branch_data is not None:
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
                            distances = np.sqrt((coords[:, 0] - r)**2 + (coords[:, 1] - c)**2)
                            if distances.min() < min_dist:
                                min_dist = distances.min()
                                closest_branch = b_idx
                        
                        if closest_branch >= 0:
                            branch_pixel_lists[closest_branch].append((r, c))
                    
                    # Update grain.PixelList to store multiple pixel lists
                    
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
                        
                    grain.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)                    
            else :
                skeleton = skeletonize(twin_mask > 0)
                true_indices = np.where(skeleton == True)
                if len(true_indices[0]) > 1:
                    grain.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)    
                elif (len(grain.SkeletonCoord) == 0) and len(true_indices[0]) == 1:
                    grain.SkeletonCoord = true_indices
                
    return Grains
    
def decompose_twins_grains_2(Grains, image, l1, std):
    
    for pt, grain in enumerate(Grains):
    
        twin_mask = np.zeros(np.flip(image.shape), dtype=np.uint8)
        
        if grain.IsTwin:
            if grain.ID == 13:
                pttt = 1
                
            # Create mask from PixelList
        
            for (x, y) in grain.PixelList:
                try:
                    twin_mask[x, y] = 1
                except IndexError:
                    continue
            if grain.ID == 37:
                titit = 1
                
           
            skeleton = skeletonize(twin_mask > 0)

            skeleton, branch_data, split = decompose_twins_2(skeleton, l1, std)
            if branch_data is not None:
                if len(branch_data) > 1 and split == 1:
                    
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
                            distances = np.sqrt((coords[:, 0] - r)**2 + (coords[:, 1] - c)**2)
                            if distances.min() < min_dist:
                                min_dist = distances.min()
                                closest_branch = b_idx
                        
                        if closest_branch >= 0:
                            branch_pixel_lists[closest_branch].append((r, c))
                    
                    # Update grain.PixelList to store multiple pixel lists
                    
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
                skeleton = skeletonize(twin_mask > 0)
                true_indices = np.where(skeleton == True)
                if len(true_indices[0]) > 1:
                    grain.SkeletonCoord = Skeleton(skeleton).path_coordinates(0)    
                elif (len(grain.SkeletonCoord) == 0) and len(true_indices[0]) == 1:
                    grain.SkeletonCoord = true_indices
                
    return Grains    

def find_overlapping_grains(grains):
    overlapping_grains = []
    n = len(grains)
    ID_grains = []

    for i in range(n):

            
        grainIPixels = []
        grainIPixels = set(map(tuple, grains[i].PixelList))
        for j in range(n):
            if j != i:
                if grains[i].ID == 108 and grains[j].ID == 135:
                    
                    x_coords = [pixel[0] for pixel in grains[i].PixelList]
                    y_coords = [pixel[1] for pixel in grains[i].PixelList]
                    x_coords1 = [pixel[0] for pixel in grains[j].PixelList]
                    y_coords1 = [pixel[1] for pixel in grains[j].PixelList]                    
                    # Plot the pixels
                    '''
                    plt.scatter(x_coords, y_coords, s=1) 
                    plt.scatter(x_coords1, y_coords1, s=1, c= 'red')# s=1 sets the size of the markers to 1

                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                    plt.grid(True)
                    plt.show()
                    '''
                    pt = 1
                grainJPixels = set(map(tuple, grains[j].PixelList))
                if len(grainIPixels.intersection(grainJPixels)) != 0 :
                    toto = grainIPixels.intersection(grainJPixels)
                    overlapping_grains.append((grains[i], grains[j]))
                    ID_grains.append((i,j,len(grainIPixels.intersection(grainJPixels))))
    return overlapping_grains, ID_grains

def remove_overlapping_pixels_ATRISK(grain1, grain2, size):
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
    
    # Plot the mask
    '''
    plt.imshow(mask, cmap='gray')
    plt.title('Updated Grain')
    plt.show()
    '''
        
    return lower_conf_grain

def handle_overlapping_grains(overlapping_grains,size):
    
    for grain1, grain2 in overlapping_grains:
        remove_overlapping_pixels(grain1, grain2, size)
        
def image_size(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2] 
    return image_rgb,(width, height)

def get_max_size(image_paths):
    """Find the maximum width and height across all images."""
    max_width, max_height = 0, 0
    for path in image_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        max_width = max(max_width, w)
        max_height = max(max_height, h)
    return max_width, max_height

def pad_image_to_max_size(image, max_size):
    """Resize an image with padding to match max dimensions."""
    h, w = image.shape[:2]
    max_w, max_h = max_size

    # Compute padding values
    top = (max_h - h) // 2
    bottom = max_h - h - top
    left = (max_w - w) // 2
    right = max_w - w - left

    # Pad the image (black background)
    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img


def read_contours_normal(txt_file, size, image_path):
    contours = []
    contours2 = []
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to RGB for proper matplotlib display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(txt_file, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split()
                points = parts[1:]  # Skip the first element as it is just a label

                if len(points) > 0:
                    # Convert normalized coordinates to image coordinates
                    contour = [(float(points[i]) * (size[0]-1)+1, float(points[i + 1]) * (size[1]-1)+1)  for i in range(0, len(points), 2)]
                    
                    # Convert to numpy array (float to keep precision)
                    contour_array = np.array(contour, dtype=np.int32)
                    contours.append(contour_array)

    # Draw contours on the padded image


    return contours


def read_contours(txt_file, size, mojo):
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
                    
#                for i in range(0 ,len(points), 2):
#                    if float(points[i]) * size[0] == width or float(points[i+1]) * size[1] == height:
#                        pt = 1
                if len(points) > 0:
                    contour = [(float(points[i]) * (size[0]), float(points[i + 1]) * (size[1])) for i in range(0, len(points), 2)]
                    contours.append(np.array(contour))

    return contours, confidences
