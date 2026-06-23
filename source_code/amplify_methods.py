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
    """
    Classify a twin grain's relationship to its neighbours and determine its twin type.

    Parameters
    ----------
    grain : Grain
        The twin grain being analysed.
    grains : list of Grain
        Full list of grains, used to look up neighbour grains by ID.
    image_shape : tuple of int
        Shape of the working image, passed through for skeleton extraction.
    Zfinal : ndarray
        Label image mapping pixels to grain IDs (not directly used here but
        kept for signature consistency with related functions).
    Average_Size : float
        Average grain size, kept for signature consistency with related
        functions (not used directly in this function body).
    background : ndarray, optional
        Background image for diagnostic plotting (unused here).

    Returns
    -------
    grain : Grain
        The input grain, possibly updated with TwinType and MisOrientation.
    type_issue : int
        Classification status code:
        -1 = not yet determined, 0 = no neighbours found, 1 = ambiguous
        (multiple neighbours on one side), 2 = neighbours found but not
        confirmed as parents, otherwise the type_error code returned by
        check_twin_type.

    Purpose
    -------
    Builds the grain's skeleton, finds its branch endpoints, and projects
    each neighbouring grain's centroid onto the skeleton's main axis to
    classify it as a "left" or "right" neighbour. Depending on how many
    neighbours fall on each side, attempts to identify the twin's parent
    grain(s) and determine whether it is a tension or compression twin via
    check_twin_type.
    """

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

    # Classify each neighbour as left/right of the twin's main skeleton axis
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
        # Exactly one neighbour on each side: check if they share the same
        # orientation (i.e. are the same "parent" grain split by the twin)
        Azimuth_P, Incli_P, Parents = check_friends(neighs_left[0], neighs_right[0])

        if Parents == True:
            grain, miso, type_error = check_twin_type(grain, Azimuth_P, Incli_P)
            grain.MisOrientation = miso
            type_issue = type_error
        else:
            type_issue = 2

    elif (len(neighs_left) == 0 and len(neighs_right) == 1) or (len(neighs_left) == 1 and len(neighs_right) == 0):
        # Only one side has a neighbour; use its orientation directly as the parent
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
        # Ambiguous: more than one neighbour on a side
        type_issue = 1

    elif (len(neighs_left) == 0 and len(neighs_right) == 0):
        # No neighbours found at all
        type_issue = 0

    return grain, type_issue

def separate_twin(grain, neighs_left, neighs_right, image_shape, maxID, skeleton_grains):
    """
    Split a twin grain that bridges two neighbours into separate grain regions.

    Parameters
    ----------
    grain : Grain
        The twin grain being split.
    neighs_left : Grain
        Neighbour grain considered as one "parent" (e.g. "Dad").
    neighs_right : Grain
        Neighbour grain considered as the other "parent" (e.g. "Mum").
    image_shape : tuple of int
        Shape of the working image, used to size masks.
    maxID : int
        Current maximum grain ID in use, so new grains get unique IDs.
    skeleton_grains : ndarray
        Skeleton image of all grains; updated in place with the new contours
        drawn by this function.

    Returns
    -------
    the_goat : Grain
        The grain representing the region where the two parent neighbours'
        convex hull overlaps the original twin grain — treated as the main
        re-derived twin grain, with Dad/Mum/Neighbours set.
    new_granulo : list of Grain
        Any additional grain fragments (size >= 10 pixels) derived from the
        parts of the twin grain outside the overlap region.

    Purpose
    -------
    Builds the convex hull of the combined pixel lists of the two neighbour
    grains, intersects it with the twin grain's own pixels to find the
    "shared" region, and creates a new Grain object for that overlap (tagged
    with both parents). Any leftover twin pixels outside the overlap are
    turned into additional new grains. Used when a single detected twin
    region actually spans across the boundary into two distinct neighbour
    grains and needs to be split accordingly.
    """
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

    # Build the contour for the overlap region (the "main" split twin)
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

    # Build contours for whatever twin pixels remain outside the overlap region
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
            # Only keep leftover fragments large enough to be meaningful grains
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
    """
    Find the highest grain ID currently in use.

    Parameters
    ----------
    grains : list of Grain
        List of Grain objects, each with an ID attribute.

    Returns
    -------
    max_ID : int
        The largest ID found among the grains (0 if the list is empty).

    Purpose
    -------
    Used to generate unique IDs for newly created grains (e.g. when
    splitting twins) without colliding with existing IDs.
    """
    max_ID = 0
    for gr in grains:
        if gr.ID > max_ID:
            max_ID = gr.ID

    return max_ID


def misorientation_angle(euler1, euler2, m, degrees=True):
    """
    Compute misorientation angle between two orientations given as Euler triplets.

    Parameters
    ----------
    euler1 : array-like of float
        First orientation as (azimuth, inclination, rotation) in degrees.
    euler2 : array-like of float
        Second orientation as (azimuth, inclination, rotation) in degrees.
    m : int
        Symmetry-equivalence index (0-15) selecting which combination of
        180-degree azimuth shifts and inclination mirroring (180 - angle)
        to apply before computing the angle. This enumerates the crystal
        symmetry-equivalent orientations so the true minimum misorientation
        can be found by calling this function for all 16 values of m.
    degrees : bool, optional
        Present for interface consistency; angles are returned in degrees
        regardless (computed via final_angle).

    Returns
    -------
    error_angle : float
        Angle (degrees) between the two orientation vectors after applying
        the symmetry operation selected by m.

    Purpose
    -------
    Applies one of 16 possible symmetry-equivalent transformations to the
    two input Euler angle sets (covering azimuth +180 degree shifts and
    inclination mirroring, individually or combined) and then computes the
    angle between the resulting orientation vectors via final_angle. Calling
    this across all m values and taking the minimum yields the true
    misorientation angle accounting for crystal symmetry.
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
    """
    Build a 3D rotation matrix about the X axis.

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    ndarray
        3x3 rotation matrix about the X axis.

    Purpose
    -------
    Used to rotate orientation vectors when converting azimuth/inclination
    angles into 3D direction vectors (see final_angle, final_angle_rot).
    """

    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotz(angle_deg):
    """
    Build a 3D rotation matrix about the Z axis.

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    ndarray
        3x3 rotation matrix about the Z axis.

    Purpose
    -------
    Used to rotate orientation vectors when converting azimuth/inclination
    angles into 3D direction vectors (see final_angle, final_angle_rot).
    """

    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


# --- Step 5: Project and classify neighbour ---
def is_projection_inside_segment(pt1, pt2, centroid, tol=1e-2):
    """
    Check whether a point's projection falls within a (slightly shortened) line segment.

    Parameters
    ----------
    pt1 : ndarray
        First endpoint of the segment.
    pt2 : ndarray
        Second endpoint of the segment.
    centroid : ndarray
        Point being tested (typically a neighbour grain's centroid).
    tol : float, optional
        Tolerance allowed below 0 when checking the projection length
        (default 1e-2).

    Returns
    -------
    bool
        True if the projection of centroid onto the segment direction falls
        between 0 (minus tol) and the segment's length, False otherwise.

    Purpose
    -------
    The segment between pt1 and pt2 is first shortened by 5% at each end
    (to avoid edge effects near the true endpoints), then centroid is
    projected onto that shortened segment's direction. Used to test whether
    a neighbouring grain's centroid lies "alongside" the twin's skeleton
    axis, as a precursor to classifying it as a left/right neighbour.
    """
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
    """
    Classify a point as lying to the left or right of a line segment.

    Parameters
    ----------
    pt1 : ndarray
        First endpoint of the reference segment.
    pt2 : ndarray
        Second endpoint of the reference segment.
    centroid : ndarray
        Point being classified (typically a neighbour grain's centroid).
    tol : float, optional
        Unused tolerance kept for signature consistency with
        is_projection_inside_segment.

    Returns
    -------
    proj_length : float
        Distance along the (shortened) segment direction where centroid
        projects onto it.
    bool
        True if centroid lies to the "left" of the segment (per the sign of
        the cross product's Z component), False if to the "right".

    Purpose
    -------
    Computes the perpendicular offset vector from the segment to centroid,
    then uses the Z component of the cross product between the segment
    direction and that offset vector to determine which side of the
    segment the point falls on. Used by grain_twin_analysis and
    find_parents_separate_twins to sort neighbouring grains into "left" and
    "right" groups relative to a twin's skeleton axis.
    """
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
    """
    Compute misorientation angles between two grains across all symmetry equivalents.

    Parameters
    ----------
    grain1 : Grain
        First grain, must have Azimuth and Inclination attributes.
    grain2 : Grain
        Second grain, must have Azimuth and Inclination attributes.

    Returns
    -------
    angles : ndarray
        Array of 16 misorientation angles (degrees), one for each symmetry
        operation enumerated by misorientation_angle.

    Purpose
    -------
    Convenience wrapper that builds Euler angle tuples from two grains'
    orientations and evaluates misorientation_angle for all 16 symmetry
    cases, returning the full array rather than just the minimum. Note: as
    written, both Euler tuples use grain2's Inclination (Phi1) — this
    matches the existing behaviour and has not been altered here.
    """
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
    """
    Determine whether a twin grain is a Tension or Compression twin.

    Parameters
    ----------
    grain : Grain
        The twin grain being classified; must have Azimuth and Inclination
        attributes.
    Azimuth_P : float
        Azimuth (degrees) of the parent grain's orientation.
    Incli_P : float
        Inclination (degrees) of the parent grain's orientation.
    Error_Angle : float, optional
        Tolerance window (degrees) around each reference angle used to
        decide a match (default 10).

    Returns
    -------
    grain : Grain
        The input grain, with TwinType set to "Tension" or "Compression" if
        exactly one type matched.
    miso : ndarray
        Array of 16 misorientation angles (one per symmetry case) between
        the twin and its parent orientation.
    type_error : int
        Status code: -1 = default/unset, 3 = no matching twin type found,
        4 = ambiguous (matched both Tension and Compression reference
        angles).

    Purpose
    -------
    Compares the twin/parent misorientation against four reference angles
    known from titanium twinning crystallography (two associated with
    Tension twins, two with Compression twins). If the misorientation falls
    within Error_Angle of exactly one twin family's reference angles, the
    grain's TwinType is set accordingly; otherwise an error code flags
    ambiguity or no match.
    """
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

    # Check whether the misorientation matches either Tension reference angle
    if np.any((miso <= angle_T1 + Error_Angle / 2) & (miso >= angle_T1 - Error_Angle / 2)):
        found_types.add("Tension")
    if np.any((miso <= angle_T2 + Error_Angle / 2) & (miso >= angle_T2 - Error_Angle / 2)):
        found_types.add("Tension")

    # Check whether the misorientation matches either Compression reference angle
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
    """
    Check whether two neighbouring grains share a common parent orientation.

    Parameters
    ----------
    studied_grain : Grain
        First neighbour grain, must have Azimuth, Inclination, size, and ID
        attributes.
    studied_grain2 : Grain
        Second neighbour grain, with the same required attributes.
    Error_Angle : float, optional
        Maximum misorientation angle (degrees) for the two grains to be
        considered the same parent orientation (default 10).

    Returns
    -------
    Azimuth : float
        Azimuth of whichever grain is larger by size.
    Inclination : float
        Inclination of whichever grain is larger by size.
    bool
        True if the minimum misorientation angle is within Error_Angle
        (i.e. the two grains are likely the same parent), False otherwise.

    Purpose
    -------
    Computes the minimum misorientation angle (mod 180) between the two
    grains across all 16 symmetry cases. If it's within tolerance, the pair
    is treated as the same parent grain (split on either side of a twin),
    and the orientation of the larger grain is returned as representative.
    Used by grain_twin_analysis and find_parents_separate_twins to confirm
    that left/right neighbours of a twin are genuinely the same parent.
    """

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
    """
    Compute the angle between two orientation vectors defined by azimuth/inclination.

    Parameters
    ----------
    azi_ejm : float
        Azimuth (degrees) of the first orientation (e.g. from optical/EJM data).
    incli_ejm : float
        Inclination (degrees) of the first orientation.
    azi_ebsd : float
        Azimuth (degrees) of the second orientation (e.g. from EBSD data).
    incli_ebsd : float
        Inclination (degrees) of the second orientation.

    Returns
    -------
    error_angle : float
        Angle (degrees) between the two resulting 3D orientation vectors.

    Purpose
    -------
    Converts each azimuth/inclination pair into a 3D unit vector by
    rotating the optical axis [0, 0, 1] first about X (inclination) then
    about Z (azimuth), then returns the angle between the two vectors via
    the dot product. This is the core angle calculation used throughout
    misorientation_angle and related comparison functions.
    """
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
    """
    Compute the angle between two orientation vectors, including an extra rotation term.

    Parameters
    ----------
    azi_ejm : float
        Azimuth (degrees) of the first orientation.
    incli_ejm : float
        Inclination (degrees) of the first orientation.
    rot_ejm : float
        Additional initial rotation (degrees) about Z for the first
        orientation, applied before the inclination/azimuth rotations.
    azi_ebsd : float
        Azimuth (degrees) of the second orientation.
    incli_ebsd : float
        Inclination (degrees) of the second orientation.
    rot_ebsd : float
        Additional initial rotation (degrees) about Z for the second
        orientation.

    Returns
    -------
    error_angle : float
        Angle (degrees) between the two resulting 3D orientation vectors.

    Purpose
    -------
    Same as final_angle, but allows an extra initial Z-rotation (rot_ejm /
    rot_ebsd) to be applied to each orientation before the standard
    inclination/azimuth rotation sequence. Used where an extra in-plane
    rotation parameter is relevant to the comparison.
    """
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

    Parameters
    ----------
    skeleton : ndarray
        Binary/uint8 skeleton image to subtract contours from.
    contours : list of ndarray
        List of (N, 2) integer contour coordinate arrays to remove.

    Returns
    -------
    skeleton : ndarray
        Copy of the input skeleton with each contour's interior filled with
        0 and its outline redrawn at value 255.

    Purpose
    -------
    Used to erase twin-grain regions from the overall grain skeleton, so
    that the remaining skeleton represents only the boundaries between
    "regular" (non-twin) grains. The outline is redrawn after filling so
    the twin boundary itself still acts as a grain boundary in the
    remaining skeleton.
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

    Parameters
    ----------
    skeleton : ndarray
        Binary/uint8 skeleton image where grain boundaries are traced.
    pad : int, optional
        Number of pixels of constant-1 padding to add around the skeleton
        before processing (default 0), useful to ensure grains touching the
        image edge are still closed regions.
    dilation_kernel : tuple of int, optional
        Kernel size used to dilate the skeleton before labelling regions
        (default (3, 3)), which helps close small gaps in the boundary.
    contour_shift : tuple of int, optional
        Offset (dx, dy) subtracted from extracted contour coordinates,
        typically used to undo the padding applied via `pad` (default (0, 0)).
    start_id : int, optional
        ID assigned to the first extracted grain; subsequent grains get
        consecutive IDs (default 1).
    mark_twinning : bool, optional
        If True, flags every extracted grain as a twin via is_twinning(True)
        (default False).

    Returns
    -------
    grains : list of Grain
        List of Grain objects, one per labelled region found in the
        (dilated, inverted) skeleton, excluding any region with no interior
        points.

    Purpose
    -------
    Dilates the skeleton boundaries slightly, labels the connected regions
    of background (i.e. the grain interiors) using the inverted dilated
    mask, and builds a Grain object for each region from its contour and
    interior pixel list. This is the core routine that turns a boundary
    skeleton image into a structured list of grains.
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

    Parameters
    ----------
    binary_img : ndarray
        Input image, any dtype convertible to boolean (nonzero = foreground).

    Returns
    -------
    ndarray
        Skeletonized image as uint8, with skeleton pixels set to 255 and
        background to 0.

    Purpose
    -------
    Thin wrapper around skimage's skeletonize that returns a displayable/
    OpenCV-compatible uint8 image instead of a boolean array, for use with
    OpenCV drawing and contour-finding functions elsewhere in this module.
    """
    ske = skeletonize(binary_img.astype(bool))
    return (ske.astype(np.uint8) * 255)

def delete_small_twins(Grains):
    """
    Remove twin grains below a minimum pixel-count threshold.

    Parameters
    ----------
    Grains : list of Grain
        List of Grain objects, some flagged as twins (IsTwin).

    Returns
    -------
    Grains : list of Grain
        The input list with any twin grain of size <= 5 pixels removed.

    Purpose
    -------
    Filters out spurious tiny twin detections (5 pixels or fewer) that are
    unlikely to represent real microstructural twins, cleaning up the grain
    list before further analysis.
    """
    index = []

    for i, gr in enumerate(Grains):
        if gr.IsTwin == True:
            if gr.size <= 5:
                index.append(i)

    for m in index:
        Grains.pop(m)

    return Grains

def poly_line(gr):
    """
    Build a valid Shapely geometry from a grain's contour points.

    Parameters
    ----------
    gr : Grain
        Grain object with a ContourPoints attribute (an (N, 2) array of
        coordinates).

    Returns
    -------
    geom : shapely.geometry.base.BaseGeometry
        A Polygon if 3+ contour points are available, a LineString for 2
        points, a Point for 1 point, or an empty Point if there are no
        points. The geometry is repaired (made valid) if necessary.

    Purpose
    -------
    Converts a grain's raw contour point array into a proper Shapely
    geometry suitable for spatial operations (area, intersection, spatial
    indexing via STRtree). Self-intersecting or otherwise invalid polygons
    are repaired using make_valid (or buffer(0) as a fallback for older
    Shapely versions) so downstream geometric operations don't fail.
    """
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

    Parameters
    ----------
    grains : list of Grain
        List of Grain objects, each expected to have an Inclination
        attribute.

    Returns
    -------
    None
        Prints a success message if all grains pass; raises ValueError on
        the first invalid grain found.

    Purpose
    -------
    Sanity-check run after orientation estimation (e.g. after
    Peaks_Optimized) to catch grains whose Inclination ended up as None,
    non-numeric, or non-finite before they propagate further into the
    pipeline and cause harder-to-diagnose errors downstream.
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
    """
    Compute per-grain mean grayscale intensity across a stack of orientation images.

    Parameters
    ----------
    grain_stats : list of Grain
        List of Grain objects, each with a PixelList attribute giving
        (x, y) pixel coordinates belonging to that grain.
    folder : str
        Path to a folder of PNG images (one per acquisition orientation),
        sorted numerically by filename.

    Returns
    -------
    grain_stats : list of Grain
        The input list, with each grain's GrayMean (array of per-image mean
        intensities) and Position (mean pixel coordinates) updated.
    gray_mean_array.mean(axis=0) : ndarray
        Empty array if no files are found; otherwise the average (across
        all grains) of mean intensity for each image in the stack.

    Purpose
    -------
    For every grain, precomputes its pixel index arrays (clipped to image
    bounds), then loops over every grayscale image in the folder computing
    the mean intensity at those pixel locations. This produces, for each
    grain, a profile of mean brightness as a function of acquisition
    orientation, the raw data later used by Peaks_Optimized to estimate
    crystallographic orientation.
    """
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
    """
    Prompt the user to select the folder containing orientation images.

    Parameters
    ----------
    original_path : object
        Unused parameter, kept for call-site signature compatibility.
    current_directory : object
        Unused parameter, kept for call-site signature compatibility.

    Returns
    -------
    file_paths : str
        Path to the selected folder.

    Raises
    ------
    ValueError
        If no folder is selected, or if the selected folder contains no
        PNG images.

    Purpose
    -------
    Opens a Tkinter directory-selection dialog (with a hidden root window)
    so the user can pick the folder of per-orientation grayscale images
    used throughout the rest of the pipeline (e.g. by gray_mean).
    """

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
    """
    Build a boolean mask of pixels matching a target color within a tolerance.

    Parameters
    ----------
    image : ndarray
        Image array of shape (H, W, 3).
    color : array-like
        Target RGB/BGR color to match, shape (3,).
    tol : int, optional
        Maximum per-channel absolute difference allowed for a match
        (default 10).

    Returns
    -------
    ndarray of bool
        Boolean mask of shape (H, W), True where all channels are within
        tol of the target color.

    Purpose
    -------
    Simple utility for isolating pixels of a specific color in pseudocolour
    or classification overlay images, e.g. to extract just the "Tension" or
    "Compression" twin coloring.
    """
    return np.all(np.abs(image - color) <= tol, axis=2)

def extract_number2(file_name):
    """
    Extract the first integer found in a filename, for use as a sort key.

    Parameters
    ----------
    file_name : str
        Path or filename, e.g. "5_Input_Test.png".

    Returns
    -------
    int
        The first integer found in the basename of file_name, or
        float('inf') if no digits are found (sorting such files last).

    Purpose
    -------
    Used as the `key` function in sorted()/glob calls throughout this module
    to ensure orientation images are processed in correct numeric order
    (e.g. "2_..." before "10_..."), rather than the lexicographic order
    that plain string sorting would give.
    """
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

    Parameters
    ----------
    grains : list of Grain
        List of Grain objects, each expected to have a PixelList attribute.

    Returns
    -------
    None
        Prints a success message if all grains pass; raises ValueError on
        the first invalid grain found.

    Purpose
    -------
    Sanity-check run at several points in the pipeline to catch grains
    whose PixelList has drifted into a non-integer or non-finite dtype
    (which would silently corrupt downstream pixel-indexing operations)
    before they cause harder-to-diagnose errors.
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
    """
    Build a skeletonized mask for a grain, from its pixel list or precomputed skeleton.

    Parameters
    ----------
    grain : Grain
        Grain object with PixelList and SkeletonCoord attributes.
    image_shape : tuple of int
        Shape of the mask to create.

    Returns
    -------
    skeleton : ndarray of bool
        Skeletonized boolean mask of the grain.

    Purpose
    -------
    If the grain doesn't yet have a precomputed SkeletonCoord, builds a mask
    from its full PixelList and skeletonizes it. Otherwise, re-skeletonizes
    using the already-stored SkeletonCoord (typically faster/cleaner since
    it avoids re-deriving the skeleton from the full grain area). Used as
    the entry point for branch/endpoint analysis on twin grains.
    """
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
    """
    Build a skan Skeleton object and summarize its branches.

    Parameters
    ----------
    skeleton : ndarray of bool
        Skeletonized mask.

    Returns
    -------
    skel : skan.Skeleton
        The skan Skeleton object wrapping the input mask, giving access to
        node coordinates and branch topology.
    branches : pandas.DataFrame
        Branch summary table (one row per branch) as returned by
        skan.summarize.

    Purpose
    -------
    Thin wrapper around skan's Skeleton/summarize calls, used wherever
    branch endpoint coordinates are needed (see
    get_branch_endpoints_centroid).
    """
    skel = Skeleton(skeleton)
    branches = summarize(skel, separator='_')
    return skel, branches


# --- Step 3: Get branch endpoints ---
def get_branch_endpoints_centroid(skel, branches, grain):
    """
    Extract branch endpoint coordinates and the grain's centroid.

    Parameters
    ----------
    skel : skan.Skeleton
        Skeleton object as returned by get_branches, providing node
        coordinates.
    branches : pandas.DataFrame
        Branch summary table with source/destination node indices in
        columns 1 and 2.
    grain : Grain
        Grain object with a Centroid attribute.

    Returns
    -------
    endpoints : list of tuple of ndarray
        List of (pt1, pt2) coordinate pairs, one per branch, giving the
        source and destination node coordinates of that branch.
    centroid_int : ndarray
        The grain's centroid, transposed and cast to int64.

    Purpose
    -------
    Converts the skan branch table's source/destination node indices into
    actual pixel coordinates via skel.coordinates, and prepares the grain's
    centroid in a matching integer format. The resulting endpoints list is
    used as the reference axis for classifying neighbouring grains as
    left/right (see is_left_or_right).
    """
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
    """
    Compute the pixel-mean centroid for a set of grains by ID.

    Parameters
    ----------
    grain_ids : iterable of int
        IDs of the grains whose centroids should be computed.
    grains : list of Grain
        Full list of Grain objects to search.

    Returns
    -------
    centroids : dict
        Mapping from grain ID to its centroid (mean of PixelList, as an
        ndarray), for every grain in grains whose ID is in grain_ids.

    Purpose
    -------
    Used by grain_twin_analysis and find_parents_separate_twins to get the
    spatial centroid of each neighbour grain, which is then projected onto
    a twin grain's skeleton axis to classify it as left/right.
    """
    centroids = {}
    for gr in grains:
        if gr.ID in grain_ids:
            pixels = np.array(gr.PixelList)
            centroids[gr.ID] = np.mean(pixels, axis=0)
    return centroids

def check_twins(Grains, list_grains):
    """
    Remove twin grain IDs from a list of neighbour IDs.

    Parameters
    ----------
    Grains : list of Grain
        Full list of Grain objects to search.
    list_grains : list of int
        List (or set) of grain IDs to filter, typically a grain's neighbour
        ID list.

    Returns
    -------
    list_grains : list of int
        The input list, with the ID of any grain flagged IsTwin removed.

    Purpose
    -------
    Used when assigning neighbour relationships (see
    get_optimized_neighbours) so that twin grains themselves are not
    counted as "neighbours" for the purposes of orientation comparison —
    only true grain-to-grain boundaries matter for that analysis.
    """
    for grain in Grains:

        if grain.ID in list_grains:

            if grain.IsTwin == True:
                list_grains.remove(grain.ID)

    return list_grains

def get_optimized_neighbours(Grainstats, image_input):
    """
    Compute pixel-adjacency-based neighbour relationships for all grains.

    Parameters
    ----------
    Grainstats : list of Grain
        List of Grain objects, each with PixelList, ID, IsTwin, and
        HaveFriends attributes.
    image_input : ndarray or tuple of int
        Either an image array (its .shape[:2] is used) or a shape tuple
        directly, defining the working image dimensions.

    Returns
    -------
    Grainstats : list of Grain
        The input list, with each grain's Neighbours attribute updated
        (only for grains flagged IsTwin or HaveFriends; others get an empty
        list).

    Purpose
    -------
    Rasterizes every grain's PixelList into a label image (z_final), then
    finds all pairs of differing, nonzero labels that are adjacent
    horizontally or vertically (a fast vectorized grid-shift approach
    rather than per-pixel neighbour checks). Builds an adjacency dictionary
    from these pairs, then assigns each twin/HaveFriends grain its set of
    neighbour IDs (excluding other twins, via check_twins).
    """
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
    """
    Identify parent grains for each twin and split twins spanning two parents.

    Parameters
    ----------
    grains : list of Grain
        List of Grain objects, some flagged as twins (IsTwin).
    image_shape : tuple of int
        Shape of the working image, used for skeleton extraction and
        twin-splitting masks.
    Zfinal : ndarray
        Label image mapping pixels to grain IDs (not directly used in this
        function body; kept for signature consistency with related
        functions).
    Average_Size : float
        Average grain size (not used directly in this function body; kept
        for signature consistency).
    skeleton_grains : ndarray
        Skeleton image of all grains, passed through to separate_twin where
        it is updated in place with new contour drawings.
    background : ndarray, optional
        Background image for diagnostic plotting (unused here).

    Returns
    -------
    grains : list of Grain
        The input list, with each twin grain's Dad/Mum attributes set where
        a single parent pair was identified, and with new grain objects
        appended (or existing ones replaced) where a twin spanning two
        distinct parent grains was split via separate_twin.

    Purpose
    -------
    For every twin grain, projects its neighbouring grains' centroids onto
    its skeleton axis to classify them as left/right (mirroring
    grain_twin_analysis), then:
    - if exactly one neighbour exists on each side and they are confirmed to
      be the same parent (via check_friends), records Dad/Mum directly;
    - if only one side has a neighbour, uses it directly as the (single)
      parent;
    - if there are multiple neighbours on both sides (in equal numbers),
      sorts them by projection distance, takes the closest pair, confirms
      they're the same parent, and if so calls separate_twin to split the
      twin region between the two parents, appending any resulting grain
      fragments.
    """
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

            # Classify each neighbour as left/right of the twin's skeleton axis
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
                # One neighbour on each side: confirm they're the same parent
                # and record Dad/Mum
                Azimuth_P, Incli_P, Parents = check_friends(neighs_left[0], neighs_right[0])
                if Parents == True:
                    grain.Dad = neighs_left[0].ID
                    grain.Mum = neighs_right[0].ID
            elif (len(neighs_left) == 0 and len(neighs_right) == 1) or (
                    len(neighs_left) == 1 and len(neighs_right) == 0):
                # Only one side has a neighbour: use it as the single parent
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
                # Multiple neighbours on both sides, equal counts: sort by
                # projection distance and try the closest pair as parents
                neighs_left = [val for _, val in sorted(zip(neighs_left_length, neighs_left))]
                neighs_right = [val for _, val in sorted(zip(neighs_right_length, neighs_right))]
                neighs_left_length = [val for _, val in sorted(zip(neighs_left_length, neighs_left_length))]
                neighs_right_length = [val for _, val in sorted(zip(neighs_right_length, neighs_right_length))]

                Azimuth_P, Incli_P, Parents = check_friends(neighs_left[0], neighs_right[0])
                if Parents == True:
                    # Confirmed two distinct parents: split the twin grain
                    # between them
                    the_goat, new_granulo = separate_twin(grain, neighs_left[0], neighs_right[0], image_shape,
                                                          find_max_ID(grains), skeleton_grains)
                    if new_granulo:
                        for gr in new_granulo:
                            grains.append(gr)
                    grains[i] = the_goat
    return grains


def measure_contour_length(contour):
    """
    Measure the total perimeter length of a closed contour.

    Parameters
    ----------
    contour : ndarray
        (N, 2) array of contour point coordinates.

    Returns
    -------
    length : float
        Sum of Euclidean distances between consecutive contour points
        (not including the final closing segment back to the first point,
        despite that segment being computed — see Purpose).

    Purpose
    -------
    Sums the Euclidean distance between each consecutive pair of contour
    points to approximate the contour's perimeter. Note: the closing
    distance between the last and first point is computed but not added to
    `length` before it is returned — this matches the existing behaviour
    and has been left unchanged here.
    """
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
    """
    Compute per-grain mean grayscale intensity, using skeleton pixels for twins.

    Parameters
    ----------
    grain_stats : list of Grain
        List of Grain objects; twin grains use SkeletonCoord pixels, others
        use the full PixelList.
    folder : str
        Path to a folder of PNG images (one per acquisition orientation).
    brightness : float
        Brightness adjustment percentage applied to every image before
        sampling (see adjust_brightness_contrast).
    contrast : float
        Contrast adjustment percentage applied to every image before
        sampling.
    plot : bool, optional
        If True, displays each orientation image with the sampled grain
        pixels overlaid as scatter points (default False).
    correction_method : bool, optional
        If True, applies an additional CLAHE + contrast + normalization
        pipeline to each image before sampling (default False).

    Returns
    -------
    grain_stats : list of Grain
        The input list, with each grain's Position and GrayMean updated.
    result : ndarray
        Mean (across all grains) intensity for each image in the stack.

    Purpose
    -------
    Same general purpose as gray_mean, but specialised for the twin-analysis
    stage: twin grains are sampled using their (thin) SkeletonCoord pixels
    rather than their full pixel area, since twin boundaries/cores are more
    representative of the underlying orientation than the full grain area
    once twins have been further subdivided. Optionally applies
    brightness/contrast and CLAHE-based normalization before sampling, and
    can visualise sampled pixels per image for debugging.
    """
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
    """
    Full pipeline: extract grains and twins, classify twin types, and render results.

    Parameters
    ----------
    orientation_path : str
        Folder containing the per-orientation grayscale image stack used to
        estimate crystallographic orientation for every grain.
    grains_image_path : str
        Path to the grain-boundary segmentation image (e.g. model output
        marking grain boundaries).
    twins_image_path : str
        Path to the twin-boundary segmentation image (model output marking
        twin boundaries).
    image_name : str
        Identifier/path for the image being processed (kept for signature
        consistency; not directly used in the body beyond assignment to
        img_study).

    Returns
    -------
    final_image : ndarray
        BGR image with grain and twin contours drawn (color-coded by twin
        type: green = Tension, blue = Compression, red = unclassified twin,
        black = normal grain boundary), flipped/rotated to the original
        orientation.
    FinalPlot_rgb : ndarray
        RGB orientation colour map image, combining the grain orientation
        colouring (from Grain_Orientation.grain_orientation) with the
        dilated contour overlay, rotated/flipped to match final_image's
        orientation.

    Purpose
    -------
    This is the top-level driver for the whole grain/twin analysis pipeline.
    At a high level it:
    1. Loads the grain and twin boundary segmentation images and
       skeletonizes both.
    2. Extracts twin grain objects directly from the dilated twin skeleton,
       then removes the twin regions from the grain skeleton so grains can
       be extracted cleanly from what remains.
    3. Extracts grain objects from the remaining skeleton.
    4. Uses a spatial index (STRtree) to find which grains overlap which
       twins above a near-total overlap threshold, flagging those grains
       as twins.
    5. Repeatedly estimates per-grain orientation from the orientation
       image stack (gray_mean + Peaks_Optimized), checks orientation
       integrity (check_peaks), and computes neighbour relationships
       (get_optimized_neighbours).
    6. Finds "friend" grains (very similar orientation) among each twin's
       neighbours and records those relationships.
    7. Resolves any pixel overlaps between at-risk grains.
    8. Rebuilds a clean label image (grain_matrix2) from the resolved
       grains, fills in any leftover unlabelled regions as new grains, and
       recomputes smoothed/dilated contours for every grain.
    9. Filters out very small grains, decomposes twins into branches
       (decompose_twins_grains / _2), identifies parent grains for twins
       and splits any twin spanning two distinct parents
       (find_parents_separate_twins), and classifies each twin's type
       (grain_twin_analysis -> check_twin_type).
    10. Marks parent grains (IsParents) for any twin successfully typed as
       Tension or Compression.
    11. Renders the final contour overlay image (color-coded by twin type)
       and a full orientation colour map, both transformed back to the
       original image orientation, and returns both.
    """


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