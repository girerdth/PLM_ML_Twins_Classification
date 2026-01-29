# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:32:20 2024

@author: ezxtg6
"""
import numpy as np
import os
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops, find_contours
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from skimage.draw import polygon
import copy
from source_code.Grain_functions import read_contours, read_contours_normal
from ultralytics import YOLO
import shutil
import random
from source_code.pseudoimage import apply_clahe, adjust_contrast, normalize_images
import math
import copy
# %%
def plot_bounding_boxes(im_input, contour_points_grain_twin, dilated):
    """
    Plots bounding boxes on the input image.

    Args:
        im_input (numpy.ndarray): The input image.
        contour_points_grain_twin (list of tuples): Bounding boxes [(x_center, y_center, width, height)] for grain twin.
        contour_points_grain_unknown (list of tuples): Bounding boxes [(x_center, y_center, width, height)] for grain unknown.
        dilated (numpy.ndarray): The dilated image (e.g., binary mask).

    Returns:
        None
    """
    # Convert image to BGR for OpenCV rectangle drawing
    image_with_boxes = im_input.copy()
    image_with_boxes = np.clip(image_with_boxes, 0, 255).astype(np.uint8)

    # Helper function to convert center-width-height to corners
    def bbox_cwh_to_corners(bbox):
        x_center, y_center, width, height = bbox
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        return x_min, y_min, x_max, y_max

    # Draw bounding boxes for grain twin
    for bbox in contour_points_grain_twin:
        x_min, y_min, x_max, y_max = bbox_cwh_to_corners(bbox)
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red rectangle

    # Draw bounding boxes for grain unknown
  #  for bbox in contour_points_grain_unknown:
  #      x_min, y_min, x_max, y_max = bbox_cwh_to_corners(bbox)
 #       cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue rectangle

    # Plot the image with bounding boxes
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for plotting
    plt.axis('off')
    plt.title("Bounding boxes for ML Model")
    
    # Color dilated image
    plt.subplot(1, 2, 2)
    plt.imshow(dilated)  # Plot color image directly
    plt.axis('off')
    plt.title("Identified features")
    
    plt.tight_layout()
    save_path = 'Structure.png'
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution
        print(f"Plot saved at: {save_path}")
    
    # Show the plot
    plt.show()




def plot_contour_manual(im_input, contour_points_grain_twin, final_grain_twin):
    """
    This function overlays multiple contours (twins) on the image (im_input).
    It draws each contour from `contour_points_grain_twin` over `im_input`.
    """

    # Step 1: Check if any contours exist
    if len(contour_points_grain_twin) == 0:
        print("No contours found.")
        return  # Exit the function if no contours

    # Step 2: Draw all contours on the image
    # Iterate through each contour in the list and draw it on the image
    for contour in contour_points_grain_twin:
        # Ensure the contour is in the correct format (np.int32) for OpenCV
        contour = np.array(contour, dtype=np.int32)
        
        # Draw the contour on the image in green (you can change the color and thickness as needed)
        cv2.drawContours(im_input, [contour], -1, (0, 255, 0), 2)  # Green color, thickness 2
    
    # Step 3: Superimpose the mask (final_grain_twin) on the image
    # Convert the mask (final_grain_twin) to a 3-channel image for blending
    mask_colored = cv2.cvtColor(final_grain_twin, cv2.COLOR_GRAY2BGR)
    
    # Ensure both the image and the mask are of the same type (uint8)
    im_input = np.array(im_input, dtype=np.uint8)
    mask_colored = np.array(mask_colored, dtype=np.uint8)
    
    # Superimpose the mask on the image to highlight the regions of interest
    blended_image = cv2.addWeighted(im_input, 1.0, mask_colored, 0.5, 0)  # Adjust transparency with 0.5
    
    # Step 4: Display the result with contours and mask
    cv2.imshow("Image with Contours", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the resulting image with contours and mask superimposed
    output_image_path = 'output_image_with_contours.png'
    cv2.imwrite(output_image_path, blended_image)
    print(f"Output image saved at: {output_image_path}")

# Example of calling the function within your existing workflow
# Assuming im_input, contour_points_grain_twin, and final_grain_twin are already defined:
# plot_contour_manual(im_input, contour_points_grain_twin, final_grain_twin)
    
def find_contour_final_test(binary_image,im_input):
    
    
    """
    Finds the bounding boxes of all contours in the given binary image.

    Args:
        binary_image (numpy.ndarray): Input binary image.

    Returns:
        bounding_boxes (list of tuples): List of bounding boxes [(x_min, y_min, x_max, y_max)] for each contour.
        contour_image (numpy.ndarray): Binary image with bounding boxes drawn.
    """
    # Invert binary image for processing
    inverted_image = ~binary_image
    skeleton = np.uint8(skeletonize(inverted_image))

    # Pad the skeleton to handle edges
    skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Generate labeled regions
    bool_ske = ~dilated_image.astype(bool)
    labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
    contours = find_contours(bool_ske, 0.5)
    contours.pop(0)
    # Calculate bounding boxes
    bounding_boxes = []
    for contour in contours:
        x_min = int(np.min(contour[:, 1]))  # X-coordinates
        y_min = int(np.min(contour[:, 0]))  # Y-coordinates
        x_max = int(np.max(contour[:, 1]))
        y_max = int(np.max(contour[:, 0]))
        bounding_boxes.append((x_min, y_min, x_max, y_max))

    # Remove inner contours
    valid_bounding_boxes = []
    for idx1, bbox1 in enumerate(bounding_boxes):
        is_inner = False
        for idx2, bbox2 in enumerate(bounding_boxes):
            if idx1 != idx2:  # Avoid comparing the same contour
                if (
                    bbox1[0] >= bbox2[0] and  # x_min1 >= x_min2
                    bbox1[1] >= bbox2[1] and  # y_min1 >= y_min2
                    bbox1[2] <= bbox2[2] and  # x_max1 <= x_max2
                    bbox1[3] <= bbox2[3]      # y_max1 <= y_max2
                ):
                    is_inner = True
                    break  # Stop checking if bbox1 is already inside bbox2
        if not is_inner:
            valid_bounding_boxes.append(bbox1)

    # Compute center, width, and height for each bounding box
    box_properties = []
    for bbox in valid_bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2
        center_y = y_min + height / 2
        box_properties.append((center_x, center_y, width, height))

    # Create contour image for visualization
    contour_image = np.zeros_like(binary_image)
    for bbox in valid_bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        contour_image[y_min:y_max, x_min:x_max] = 255

    return box_properties

def find_contour_final(binary_image,mode):
    
    inverted_image = ~binary_image
    if mode == 0:
        skeleton = np.uint8(skeletonize(inverted_image))
    elif mode == 1:
        skeleton = ~inverted_image
    else:
        skeleton = ~inverted_image
   # skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1)

    # Step 3: Replace the border pixels with 1s in the skeleton

    bool_ske_final = ~dilated_image.astype(bool)     
     
    inverted_image = ~binary_image
    if mode == 0 or mode == 1:
        skeleton = np.uint8(skeletonize(inverted_image))
    else:
        skeleton = ~inverted_image
   # skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(skeleton, kernel, iterations=1)
        skeleton = np.uint8(skeletonize(dilated_image))
        
    # Step 3: Replace the border pixels with 1s in the skeleton
    if mode == 0 or mode == 2:
        skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=1)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=1) 
    
    bool_ske = ~dilated_image.astype(bool)

    if mode == 0:
        labeled_array_ske, num_features_ske = label(bool_ske, return_num=True)
    elif mode == 2:
        binary_imafe = np.pad(binary_image, pad_width=1, mode='constant', constant_values=1)
        bool_ske = binary_imafe.astype(bool)
        labeled_array_ske, num_features_ske = label(~bool_ske, return_num=True)
    elif mode == 3:
       int_image = (binary_image < 255).astype(int)
       skeleton = np.uint8(skeletonize(int_image))
       kernel = np.ones((3, 3), np.uint8)
       dilated_image = cv2.dilate(skeleton, kernel, iterations=1) 
       
       bool_ske = ~dilated_image.astype(bool)
       
       labeled_array_ske, num_features_ske = label(skeleton, return_num=True)
       
    else:
        labeled_array_ske, num_features_ske = label(~bool_ske, return_num=True)
    
    contours = find_contours(labeled_array_ske, 0.5)
    
    contour_image = np.zeros_like(binary_image)
    contour_points_list = []
    for contour in contours:
        contour_points = []
        for point in contour:
            y, x = point
            if mode == 0 or mode == 2:
                corrected_x = max(0, math.floor(x)-1) 
                corrected_y = max(0, math.floor(y)-1) 
                contour_points.append((corrected_x, corrected_y))
                
                contour_image[math.floor(y)-1, math.floor(x)-1] = 255
            else:

                corrected_x = max(0, round(x)) 
                corrected_y = max(0, round(y)) 
                contour_points.append((corrected_x, corrected_y))
                contour_image[round(y), round(x)] = 255
        contour_points_list.append(contour_points)
    pt = 1
    return contour_points_list   


def generate_random(im_input,pt,images,m1,m2,n1,n2):
    merged_img = im_input
    img_list = copy.deepcopy(images)
    if pt == 0 or pt == 4:
        img1 = cv2.imread(img_list[0*4], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_list[1*4], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(img_list[2*4], cv2.IMREAD_GRAYSCALE)
        img1_clahe = apply_clahe(img1)
        img2_clahe = apply_clahe(img2)
        img3_clahe = apply_clahe(img3)
        
        img1_contrast = adjust_contrast(img1_clahe)
        img2_contrast = adjust_contrast(img2_clahe)
        img3_contrast = adjust_contrast(img3_clahe)
        
        merged_img = cv2.merge([img3_contrast, img2_contrast, img1_contrast])
        im_input = merged_img[m1:m2,n1:n2,:] 
        
    else: 
        if pt == 1:
            pick = 4
        if pt == 2:
            pick = 9
        if pt == 3:
            pick = 7
        
        #pick = random.randint(0,len(img_list)-1)
        pick2 = (pick + int(40 / 10)) % 18
        pick3 = (pick + 2 * int(40 / 10)) % 18 
        
        img1 = cv2.imread(images[pick], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images[pick2], cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(images[pick3], cv2.IMREAD_GRAYSCALE)
        
        img1_clahe = apply_clahe(img1)
        img2_clahe = apply_clahe(img2)
        img3_clahe = apply_clahe(img3)
        
        img1_contrast = adjust_contrast(img1_clahe)
        img2_contrast = adjust_contrast(img2_clahe)
        img3_contrast = adjust_contrast(img3_clahe)
        
        merged_img = cv2.merge([img3_contrast, img2_contrast, img1_contrast])
        im_input = merged_img[m1:m2,n1:n2,:]   
    im_input = im_input.astype(np.uint8)        
    return im_input

def create_orientation_folder(srcs, orientation_folder,nbr_split, l):
    
    #images = glob.glob(os.path.join(srcs, '*Twins_vf.png'))
    images = glob.glob(os.path.join(srcs, '*.bmp'))
    img = cv2.imread(images[0])
    [m,n,d] = np.shape(img)
    size = [n,m]
   
    # Step 4: Draw twins (bounding boxes or contours)
    # Step 4: Create an empty mask (black image
    
  #grain_couple[np.all(grain_couple == black, axis=-1)] = white
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    image_label_pairs = []
    randoms = 4

        
    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])
                for tt in range(4):
                    output = f'{l}_Input'
                    m = 0
                    final = os.path.join(orientation_folder,output)
                    os.makedirs(final)
                    for imgtotal in images:
                        name = f'{m}.png'
                        img_f = cv2.imread(imgtotal)
                        im_input = img_f[m1:m2,n1:n2,:]
                        cv2.imwrite(os.path.join(final,name),im_input)
                        m = m + 10
                    l = l + 1

    return l




def create_dataset_manual_random(srcs, label_origin_file, images_folder, labels_folder, masks_folder, nbr_split, l):
    
    images = glob.glob(os.path.join(srcs, '*.bmp'))

    img = cv2.imread(images[0])
    [m,n,d] = np.shape(img)
    size = [n,m]
    twins = read_contours_normal(label_origin_file, size, images[0])

    # Step 4: Draw twins (bounding boxes or contours)
    # Step 4: Create an empty mask (black image)
    mask = np.zeros((m, n), dtype=np.uint8)  # Same size as the image, but single channel
    
    # Step 5: Draw the contours on the mask
    # Assume `twins` is a list of contours, where each contour is a list of points
    for twin in twins:
        cv2.drawContours(mask, [np.array(twin)], -1, (255), thickness=cv2.FILLED)  # White color, filled contours
    
    # Step 6: Convert the mask to a 3-channel image for blending with the original image
    twin_image = ~mask
    cv2.imwrite(os.path.join(srcs, 'label.png'), twin_image)

    train_images_folder = os.path.join(images_folder, 'train')
    val_images_folder = os.path.join(images_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')
    train_masks_folder = os.path.join(masks_folder, 'train')
    val_masks_folder = os.path.join(masks_folder, 'val')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(val_masks_folder, exist_ok=True)
        
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    image_label_pairs = []
    randoms = 4

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                twin = copy.deepcopy(twin_image)
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])
                

                
                for pt in range(randoms):
                    final_grain_twin = twin[m1:m2,n1:n2]
                    contour_points_twin = find_contour_final(final_grain_twin,1)
                    [m1n,n1n] = np.shape(final_grain_twin)
                    SizeIm = [n1n,m1n]
                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'    
                    
                    image_label_pairs.append((im_input, final_grain_twin, contour_points_twin, output_filename, txt_filename,SizeIm))
                    l += 1 
                    
                    im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                    final_grain_twin_rot = cv2.rotate(final_grain_twin, cv2.ROTATE_90_CLOCKWISE)
                    contour_points_twin_rot = find_contour_final(final_grain_twin_rot,1)
                    [m1n,n1n] = np.shape(final_grain_twin_rot)
                    SizeIm = [n1n,m1n]
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'
                    
                    image_label_pairs.append((im_input_rot, final_grain_twin_rot, contour_points_twin_rot, output_filename, txt_filename,SizeIm))
                    l += 1 
                    
                    im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    final_grain_twin_rot = cv2.rotate(final_grain_twin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    contour_points_twin_rot = find_contour_final(final_grain_twin_rot,1)
                    [m1n,n1n] = np.shape(final_grain_twin_rot)
                    SizeIm = [n1n,m1n]
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'
                    
                    image_label_pairs.append((im_input_rot, final_grain_twin_rot, contour_points_twin_rot, output_filename, txt_filename,SizeIm))
                    l += 1 
                    
                    im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                    final_grain_twin_rot = cv2.rotate(final_grain_twin, cv2.ROTATE_180)
                    contour_points_twin_rot = find_contour_final(final_grain_twin_rot,1)
                    [m1n,n1n] = np.shape(final_grain_twin_rot)
                    SizeIm = [n1n,m1n]
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'
                    
                    image_label_pairs.append((im_input_rot, final_grain_twin_rot, contour_points_twin_rot, output_filename, txt_filename,SizeIm))
                    l += 1 
                                       

        train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=32, shuffle=True)
        save_pairs_masks(train_pairs, train_images_folder, train_labels_folder, train_masks_folder)
        save_pairs_masks(val_pairs, val_images_folder, val_labels_folder, val_masks_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l

def create_dataset_manual_random_test(srcs, label_origin_file, grain_file, images_folder, labels_folder, masks_folder, nbr_split, l):
    
    images = glob.glob(os.path.join(srcs, '*.bmp'))
    
    img = cv2.imread(images[0])
    [m,n,d] = np.shape(img)
    size = [n,m]
    twins, confi = read_contours(label_origin_file, size, 1)
    
    twins_masks_folder = os.path.join(masks_folder, 'Twins')
    grains_masks_folder = os.path.join(masks_folder, 'Grains')
    both_masks_folder = os.path.join(masks_folder, 'Both')

    os.makedirs(twins_masks_folder, exist_ok=True)
    os.makedirs(grains_masks_folder, exist_ok=True)
    os.makedirs(both_masks_folder, exist_ok=True)

    # Step 4: Draw twins (bounding boxes or contours)
    # Step 4: Create an empty mask (black image)
    mask = np.zeros((m, n), dtype=np.uint8)
    twins_cop = copy.deepcopy(twins)
    
    
    for twin in twins_cop:
        twin = np.array(twin, dtype=np.int32)
        cv2.drawContours(mask, [np.array(twin)], -1, (255), thickness=cv2.FILLED)  # White color, filled contours
   
    # Step 6: Convert the mask to a 3-channel image for blending with the original image
    twin_image = ~mask
    
    grain_image = cv2.imread(grain_file)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    grain_image_grain = copy.deepcopy(grain_image)
    grain_image_grain = cv2.cvtColor(grain_image_grain, cv2.COLOR_RGB2GRAY)
    #grain_couple[np.all(grain_couple == black, axis=-1)] = white
  

        
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    image_label_pairs = []
    randoms = 4

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                twin = copy.deepcopy(twin_image)
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])
                

                
                for pt in range(randoms):
                    final_grain_twin = twin[m1:m2,n1:n2]
                    final_grain_grain = grain_image_grain[m1:m2,n1:n2]

                    contour_points_twin = find_contour_final(final_grain_twin,1)
                    contour_points_grains = find_contour_final(final_grain_grain,0)
                    [m1n,n1n] = np.shape(final_grain_twin)
                    SizeIm = [n1n,m1n]
                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'    
                    
                    image_label_pairs.append((im_input, final_grain_grain, contour_points_grains, final_grain_twin, contour_points_twin, output_filename, txt_filename,SizeIm))
                    l += 1 
                    
                    im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                    

        #train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs_tests(image_label_pairs, images_folder, labels_folder, grains_masks_folder, twins_masks_folder, both_masks_folder)
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l

def create_dataset_manual_random_EBSD(srcs, grain_file, images_folder, labels_folder, masks_folder, nbr_split, l):
    
    images = glob.glob(os.path.join(srcs, '*.bmp'))
    
    img = cv2.imread(images[0])
    [m,n,d] = np.shape(img)
    size = [n,m]
    grain_image = cv2.imread(grain_file)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    #grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_grain = copy.deepcopy(grain_image)

    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_grain[np.all(grain_image_grain == red, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == black, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)
    grain_image_grain = cv2.cvtColor(grain_image_grain, cv2.COLOR_RGB2GRAY)    
        
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    image_label_pairs = []
    randoms = 4

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])
                
                for pt in range(randoms):
                    final_grain_twin = grain_image_twin[m1:m2,n1:n2]
                    final_grain_grain = grain_image_grain[m1:m2,n1:n2]
                    

                    contour_points_twin = find_contour_final(final_grain_twin,1)
                    contour_points_grains = find_contour_final(final_grain_grain,0)
                    [m1n,n1n] = np.shape(final_grain_twin)
                    SizeIm = [n1n,m1n]
                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'    
                    
                    image_label_pairs.append((im_input, final_grain_grain, output_filename))
                    l += 1 

        #train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs_simple(image_label_pairs, images_folder)
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l

def save_pairs_simple(image_label_pairs, images_folder):
    for im_input, grain_image, output_filename in image_label_pairs:
        #print(os.path.join(images_folder, output_filename))
        cv2.imwrite(os.path.join(images_folder, output_filename), grain_image)

def get_latest_predict_dir(base_dir):
    """Get the most recent prediction directory."""
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not all_dirs:
        return None
    all_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    return os.path.join(base_dir, all_dirs[0])

def manual_semi_supervised(source_folder, destination_folder, model_path, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    images = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(images)
    img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)

    size = [n,m]
     
    images_folder = os.path.join(destination_folder, 'images')
    labels_folder = os.path.join(destination_folder, 'labels')
     
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
        
    crop = []
    sections = []

    predict_base_dir = r'D:\Twins_ML_Model\runs\segment'
    
    for i in range(1, 10):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    image_label_pairs = []
    randoms = 4
    
    image_label_pairs = []
    
    model = YOLO(model_path)

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])
                sizes = []
                for pt in range(randoms):

                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    [m1n,n1n,d] = np.shape(im_input)
                    SizeIm = [n1n,m1n]
                    sizes.append(SizeIm)
                    output_filename = f'{l}_Input.png'
                
                    cv2.imwrite(os.path.join(images_folder, output_filename), im_input)
                
                    model.predict(os.path.join(images_folder, output_filename), save=True, save_txt=True, conf=0.5, imgsz=640, max_det=3000)
                    predict_dir = get_latest_predict_dir(predict_base_dir)
                
                    labels_predict = os.path.join(predict_dir,'labels')
                
                    label_files = glob.glob(os.path.join(labels_predict, f'{l}_Input.txt'))
                    
                
                    if not label_files:
                        with open(os.path.join(labels_predict, f'{l}_Input.txt'), 'w') as file:
                            pass  # Do nothing, just create the file
                        label_files = glob.glob(os.path.join(labels_predict, f'{l}_Input.txt'))
                    
                    
               

                    l += 1 
                    for m in range(3):
                        if m == 0:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                        elif m == 1:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)   
                        elif m == 2:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                        [m1n,n1n,d] = np.shape(im_input_rot)
                        sizes.append([n1n,m1n])
                        output_filename_rot = f'{l}_Input.png'
                        cv2.imwrite(os.path.join(images_folder, output_filename_rot), im_input_rot)
                        model.predict(os.path.join(images_folder, output_filename_rot), save=True, save_txt=True, conf=0.5, imgsz=640, max_det=3000)
                        predict_dir = get_latest_predict_dir(predict_base_dir)
                        labels_predict = os.path.join(predict_dir,'labels')
                        
                        label_files = glob.glob(os.path.join(labels_predict, f'{l}_Input.txt'))
                        if not label_files:
                            with open(os.path.join(labels_predict, f'{l}_Input.txt'), 'w') as file:
                                pass  # Do nothing, just create the file
                            label_files = glob.glob(os.path.join(labels_predict, f'{l}_Input.txt'))
                        

                        l += 1 
                
                mask_all = np.zeros(np.flip(SizeIm), dtype=np.uint8)
                toto = l % 16
                titi = 0
                for el in range(l-16,l):
                    
                    mask = np.zeros(np.flip(sizes[titi]), dtype=np.uint8)
                    label_files = glob.glob(os.path.join(labels_predict, f'{el}_Input.txt'))
                    twins = read_contours(label_files[0], sizes[titi], images[0])
                    for twin in twins:
                        cv2.drawContours(mask, [np.array(twin)], -1, (255), thickness=cv2.FILLED)  # White color, filled contours
                    if titi == 0 or titi == 4 or titi == 8 or titi == 12:
                        mask_all = mask_all + mask
                    if titi == 1 or titi == 5 or titi == 9 or titi == 13:
                        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        mask_all = mask_all + mask          
                    if titi == 2 or titi == 6 or titi == 10 or titi == 14:
                        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                        mask_all = mask_all + mask   
                    if titi == 3 or titi == 7 or titi == 11 or titi == 15:
                        mask = cv2.rotate(mask, cv2.ROTATE_180)
                        mask_all = mask_all + mask   
                    titi += 1
                twin_image = ~mask_all
                titi = 0
                for el in range(l-16,l):
                    #titi = el-((toto-1)*16)-1
                    ImSize = sizes[titi]
                    if titi == 0 or titi == 4 or titi == 8 or titi == 12:
                        twin_image2 = twin_image
                    if titi == 1 or titi == 5 or titi == 9 or titi == 13:
                        twin_image2 = cv2.rotate(twin_image, cv2.ROTATE_90_CLOCKWISE)
                            
                    if titi == 2 or titi == 6 or titi == 10 or titi == 14:
                        twin_image2 = cv2.rotate(twin_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                   
                    if titi == 3 or titi == 7 or titi == 11 or titi == 15:
                        twin_image2 = cv2.rotate(twin_image, cv2.ROTATE_180)
                    contour_points_twin = find_contour_final(twin_image2,1) 
                    txt_filename = f'{el}_Input.txt'
                    with open(os.path.join(labels_folder, txt_filename), 'w') as file:
                        for contour_points in contour_points_twin:
                            file.write('1 ')
                            if isinstance(contour_points, tuple) and len(contour_points) == 4:
                                file.write(f'{contour_points[0]/ImSize[0]} {contour_points[1]/ImSize[1]} '
                                           f'{contour_points[2]/ImSize[0]} {contour_points[3]/ImSize[1]}\n')
                            elif isinstance(contour_points, list):
                                file.write(' '.join(f'{point[0]/ImSize[0]} {point[1]/ImSize[1]}' for point in contour_points))
                                file.write('\n')
                            else:
                                raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")
                    titi += 1
    return l

def create_dataset_manual_random_simple(srcs, grain_file, images_folder, nbr_split, l):
    
    images = glob.glob(os.path.join(srcs, '*.bmp'))
    
    img = cv2.imread(images[0])
    [m,n,d] = np.shape(img)
    size = [n,m]
    grain_image = cv2.imread(grain_file)
 
        
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    image_label_pairs = []
    randoms = 4

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])
                
                for pt in range(randoms):
                    grain_image2 = grain_image[m1:m2,n1:n2,:]
                    
                    [m1n,n1n,d] = np.shape(grain_image2)
                    SizeIm = [n1n,m1n]
                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'    
                    
                    image_label_pairs.append((im_input, grain_image2, output_filename))
                    l += 1 

        #train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs_simple(image_label_pairs, images_folder)
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l



def prepare_data(image_origin, images_folder, nbr_split, l):
    
    
    
    img = cv2.imread(image_origin)
    [m,n,d] = np.shape(img)
    size = [n,m]

   
   
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if  i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    image_label_pairs = []
    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                im_input[:, :, :] = img[int(i * crop[idx1][0]):int((i + 1) * crop[idx1][0] ),
                              int(j * crop[idx1][1]):int((j + 1) * crop[idx1][1]) ,:]
                
                #contour_points_grain_unknown = find_contour_final_test(final_grain_unknown,im_input)
                        
                output_filename = f'{l}_Input.png'
                
                #if l == 37:               
                #    plot_bounding_boxes(im_input,contour_points_grain_twin,final_couple)
                
                image_label_pairs.append((im_input, output_filename))
           
                l += 1 

      
        save_pairs_simple(image_label_pairs, images_folder)

        image_label_pairs = []
    
    return l
def save_pairs_both(pairs, image_folder, labels_folder):
    for im_input, contour_points_grain, contour_points_twin, output_filename, txt_filename,sizeIm in pairs:
        cv2.imwrite(os.path.join(image_folder, output_filename), im_input)
        with open(os.path.join(labels_folder, txt_filename), 'w') as file:
            for contour_points in contour_points_grain:
                file.write('1 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")
  
            for contour_points in contour_points_twin:
                file.write('2 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")

def create_dataset_both(nbr_inputs, nbr_split, source_folder, image_folder, labels_folder, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    gt_folder = os.path.join(source_folder, 'GB')
    
    data_files = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(data_files)
    gb = glob.glob(os.path.join(gt_folder, '*Twins_vf.png'))
    
    img = cv2.imread(data_files[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)
    
    train_images_folder = os.path.join(image_folder, 'train')
    val_images_folder = os.path.join(image_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')
 #   test_image_folder = os.path.join(image_folder,'test')
#    test_labels_folder = os.path.join(labels_folder,'test')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X * 2 - 1)
            sect_Y = int(n / crop_Y * 2 - 1)
            sections.append([sect_X,sect_Y])
    
    grain_path = gb[0]
    grain_image = cv2.imread(grain_path)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    #grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_grain = copy.deepcopy(grain_image)
    # Replace pixel values below 255 with 0
    
    
    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_grain[np.all(grain_image_grain == red, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == black, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)
    grain_image_grain = cv2.cvtColor(grain_image_grain, cv2.COLOR_RGB2GRAY)
    #grain_couple[np.all(grain_couple == black, axis=-1)] = white
    
    image_label_pairs = []
    # Display all grain images

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                im_input = np.zeros((crop[idx1][0], crop[idx1][1], nbr_inputs))
                for idx in range(nbr_inputs):
                    if nbr_inputs > 3:
                        shift = idx
                    else:
                        if img_total > 3:
                            shift = idx * 4
                        else:
                            shift = idx
                    img = cv2.imread(data_files[shift], cv2.IMREAD_GRAYSCALE)
                    im_input[:, :, idx] = img[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                              int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                
            
                final_grain_twin = grain_image_twin[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                final_grain_grain = grain_image_grain[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]

                contour_points_grain = find_contour_final(final_grain_grain,0)
                contour_points_twin = find_contour_final(final_grain_twin,1)
                #contour_points_grain_unknown = find_contour_final_test(final_grain_unknown,im_input)
         
                [m1,n1] = np.shape(final_grain_twin)
                SizeIm = [n1,m1]
                
                output_filename = f'{l}_Input.png'
                txt_filename = f'{l}_Input.txt'
                
                image_label_pairs.append((im_input, contour_points_grain, contour_points_twin, output_filename, txt_filename,SizeIm))
                
                grain_image[grain_image < 255] = 0

                l += 1 
                for m in range(3):
                    if m == 0:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_CLOCKWISE)
                        final_grain_rot_g = cv2.rotate(final_grain_grain, cv2.ROTATE_90_CLOCKWISE)
             
                    elif m == 1:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        final_grain_rot_g = cv2.rotate(final_grain_grain, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  
                    elif m == 2:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_180)
                        final_grain_rot_g = cv2.rotate(final_grain_grain, cv2.ROTATE_180)

                    contour_points_grain_twin = find_contour_final(final_grain_rot_t,1)
                    contour_points_grain_grain = find_contour_final(final_grain_rot_g,0)
              
                    output_filename_rot = f'{l}_Input.png'
                    txt_filename_rot = f'{l}_Input.txt'
                    [m1,n1] = np.shape(final_grain_rot_t)
                    SizeIm = [n1,m1]
                    image_label_pairs.append((im_input_rot, contour_points_grain_grain,  contour_points_grain_twin, output_filename_rot, txt_filename_rot,SizeIm))
                           
                    l += 1    

        train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs_both(train_pairs, train_images_folder, train_labels_folder)
        save_pairs_both(val_pairs, val_images_folder, val_labels_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l

def create_dataset_grain(nbr_inputs, nbr_split, source_folder, image_folder, labels_folder, masks_folder, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    gt_folder = os.path.join(source_folder, 'GB')
    
    data_files = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(data_files)
    gb = glob.glob(os.path.join(gt_folder, '*Twins_vf.png'))
    
    img = cv2.imread(data_files[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)
    
    train_images_folder = os.path.join(image_folder, 'train')
    val_images_folder = os.path.join(image_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')
    train_masks_folder = os.path.join(masks_folder, 'train')
    val_masks_folder = os.path.join(masks_folder, 'val')
 #   test_image_folder = os.path.join(image_folder,'test')
#    test_labels_folder = os.path.join(labels_folder,'test')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(val_masks_folder, exist_ok=True)   
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X * 2 - 1)
            sect_Y = int(n / crop_Y * 2 - 1)
            sections.append([sect_X,sect_Y])
    
    grain_path = gb[0]
    grain_image = cv2.imread(grain_path)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_unknown = copy.deepcopy(grain_image)
    # Replace pixel values below 255 with 0
    
    
    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_twin[np.all(grain_image_twin == red, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == blue, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)

    #grain_couple[np.all(grain_couple == black, axis=-1)] = white
    image_label_pairs = []
    # Display all grain images

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                im_input = np.zeros((crop[idx1][0], crop[idx1][1], nbr_inputs))
                for idx in range(nbr_inputs):
                    if nbr_inputs > 3:
                        shift = idx
                    else:
                        if img_total > 3:
                            shift = idx * 4
                        else:
                            shift = idx
                    img = cv2.imread(data_files[shift], cv2.IMREAD_GRAYSCALE)
                    im_input[:, :, idx] = img[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                              int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                
            
                final_grain_twin = grain_image_twin[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
               # final_grain_unknown = grain_image_unknown[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
               #                   int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                final_couple = grain_couple[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2),:]
                contour_points_grain_twin = find_contour_final(final_grain_twin,0)
       
                #contour_points_grain_unknown = find_contour_final_test(final_grain_unknown,im_input)
         
                [m1,n1] = np.shape(final_grain_twin)
                SizeIm = [n1,m1]
                
                output_filename = f'{l}_Input.png'
                txt_filename = f'{l}_Input.txt'
                
                
                #if l == 37:               
                #    plot_bounding_boxes(im_input,contour_points_grain_twin,final_couple)
                
                image_label_pairs.append((im_input, final_grain_twin, contour_points_grain_twin, output_filename, txt_filename,SizeIm))
                
                grain_image[grain_image < 255] = 0

                l += 1 
                for m in range(3):
                    if m == 0:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_CLOCKWISE)
                       # final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_CLOCKWISE)
             
                    elif m == 1:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                     #   final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  
                    elif m == 2:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_180)
                      #  final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_180)

                    contour_points_grain_twin = find_contour_final(final_grain_rot_t,0)
  
                   # contour_points_grain_unknown = find_contour_final_test(final_grain_rot_u,im_input)
              
                    output_filename_rot = f'{l}_Input.png'
                    txt_filename_rot = f'{l}_Input.txt'
                    [m1,n1] = np.shape(final_grain_rot_t)
                    SizeIm = [n1,m1]
                    image_label_pairs.append((im_input_rot, final_grain_twin, contour_points_grain_twin, output_filename_rot, txt_filename_rot,SizeIm))
                           
                    l += 1    

        train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs_masks(train_pairs, train_images_folder, train_labels_folder, train_masks_folder)
        save_pairs_masks(val_pairs, val_images_folder, val_labels_folder, val_masks_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l


def plot_images_with_contours(image, mask, contours, title="Image, Contours, and Mask"):
    """Function to plot the original image, its contours, and the final grain mask side by side."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1-row, 3-column plot layout
    
    # Ensure the image is grayscale (single channel)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image  # Already grayscale
    
    # Convert grayscale image to RGB for visualization
    image_rgb = cv2.merge([image_gray] * 3)  # Convert grayscale to 3-channel RGB
    
    # Convert contours list into OpenCV format
    processed_contours = [np.array(c, dtype=np.int32) for c in contours]  # Ensure proper format
    
    # Draw contours on the image
    contour_image = image_rgb.copy()
    cv2.drawContours(contour_image, processed_contours, -1, (0, 255, 0), 1)  # Green contours
    
    # Plot original image
    axes[0].imshow(image_gray, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot image with contours
    axes[1].imshow(contour_image)
    axes[1].set_title("Contours")
    axes[1].axis("off")

    # Plot final mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Final Mask")
    axes[2].axis("off")
    
    plt.suptitle(title)
    plt.show()
    
def create_dataset_grain_val(nbr_inputs, nbr_split, source_folder, image_folder, labels_folder, masks_folder, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    gt_folder = os.path.join(source_folder, 'GB')
    
    images = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(images)
    gb = glob.glob(os.path.join(gt_folder, '*GB.png'))
    
    img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)
    
    #train_images_folder = os.path.join(image_folder, 'train')
    val_images_folder = os.path.join(image_folder, 'val')
    #train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')
    #train_masks_folder = os.path.join(masks_folder, 'train')
    val_masks_folder = os.path.join(masks_folder, 'val')
    
    #os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    #os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    #os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(val_masks_folder, exist_ok=True)   
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    grain_path = gb[0]
    grain_image = cv2.imread(grain_path)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_unknown = copy.deepcopy(grain_image)
    # Replace pixel values below 255 with 0
    
    
    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_twin[np.all(grain_image_twin == red, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == blue, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)

    image_label_pairs = []

    randoms = 1
    
    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                twin = copy.deepcopy(grain_image_twin)
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])                
                
                
                for pt in range(randoms):
                    final_grain_twin = twin[m1:m2,n1:n2]
                    contour_points_twin = find_contour_final(final_grain_twin,0)
                    [m1n,n1n] = np.shape(final_grain_twin)
                    SizeIm = [n1n,m1n]
                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'
                    image_label_pairs.append((im_input, final_grain_twin, contour_points_twin, output_filename, txt_filename,SizeIm))
                    l += 1 
                    #plot_images_with_contours(im_input, final_grain_twin, contour_points_grain_twin)               
                    for m in range(3):
                        if m == 0:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                            final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_CLOCKWISE)
                           # final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_CLOCKWISE)
                 
                        elif m == 1:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                         #   final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_COUNTERCLOCKWISE)
                      
                        elif m == 2:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                            final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_180)
                          #  final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_180)
    
                        contour_points_grain_twin_rot = find_contour_final(final_grain_rot_t,0)
      
                   # contour_points_grain_unknown = find_contour_final_test(final_grain_rot_u,im_input)
                        #plot_images_with_contours(im_input_rot, final_grain_rot_t, contour_points_grain_twin)
                        output_filename_rot = f'{l}_Input.png'
                        txt_filename_rot = f'{l}_Input.txt'
                        [m1n,n1n] = np.shape(final_grain_rot_t)
                        SizeIm = [n1n,m1n]
                        image_label_pairs.append((im_input_rot, final_grain_rot_t, contour_points_grain_twin_rot, output_filename_rot, txt_filename_rot,SizeIm))
                               
                        l += 1    

     #   train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
     #   save_pairs_masks(train_pairs, train_images_folder, train_labels_folder, train_masks_folder)
        save_pairs_masks(image_label_pairs, val_images_folder, val_labels_folder, val_masks_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l
    
def create_dataset_grain_train(nbr_inputs, nbr_split, source_folder, image_folder, labels_folder, masks_folder, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    gt_folder = os.path.join(source_folder, 'GB')
    
    images = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(images)
    gb = glob.glob(os.path.join(gt_folder, '*GB.png'))
    
    img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)
    
    train_images_folder = os.path.join(image_folder, 'train')
    #val_images_folder = os.path.join(image_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    #val_labels_folder = os.path.join(labels_folder, 'val')
    train_masks_folder = os.path.join(masks_folder, 'train')
    #val_masks_folder = os.path.join(masks_folder, 'val')
    
    os.makedirs(train_images_folder, exist_ok=True)
    #os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    #os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    #os.makedirs(val_masks_folder, exist_ok=True)   
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X)
            sect_Y = int(n / crop_Y)
            sections.append([sect_X,sect_Y])
    
    grain_path = gb[0]
    grain_image = cv2.imread(grain_path)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_unknown = copy.deepcopy(grain_image)
    # Replace pixel values below 255 with 0
    
    
    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_twin[np.all(grain_image_twin == red, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == blue, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)

    image_label_pairs = []

    randoms = 1
    
    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                twin = copy.deepcopy(grain_image_twin)
                m1 = int(i * crop[idx1][0])
                m2 = int((i + 1) * crop[idx1][0])
                n1 = int(j * crop[idx1][1])
                n2 = int((j + 1) * crop[idx1][1])                
                
                
                for pt in range(randoms):
                    final_grain_twin = twin[m1:m2,n1:n2]
                    contour_points_twin = find_contour_final(final_grain_twin,0)
                    [m1n,n1n] = np.shape(final_grain_twin)
                    SizeIm = [n1n,m1n]
                    im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                    im_input = generate_random(im_input,pt,images,m1,m2,n1,n2)
                    
                    output_filename = f'{l}_Input.png'
                    txt_filename = f'{l}_Input.txt'
                    image_label_pairs.append((im_input, final_grain_twin, contour_points_twin, output_filename, txt_filename,SizeIm))
                    l += 1 
                    #plot_images_with_contours(im_input, final_grain_twin, contour_points_grain_twin)               
                    for m in range(3):
                        if m == 0:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                            final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_CLOCKWISE)
                           # final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_CLOCKWISE)
                 
                        elif m == 1:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                         #   final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_COUNTERCLOCKWISE)
                      
                        elif m == 2:
                            im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                            final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_180)
                          #  final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_180)
    
                        contour_points_grain_twin_rot = find_contour_final(final_grain_rot_t,0)
      
                   # contour_points_grain_unknown = find_contour_final_test(final_grain_rot_u,im_input)
                        #plot_images_with_contours(im_input_rot, final_grain_rot_t, contour_points_grain_twin)
                        output_filename_rot = f'{l}_Input.png'
                        txt_filename_rot = f'{l}_Input.txt'
                        [m1n,n1n] = np.shape(final_grain_rot_t)
                        SizeIm = [n1n,m1n]
                        image_label_pairs.append((im_input_rot, final_grain_rot_t, contour_points_grain_twin_rot, output_filename_rot, txt_filename_rot,SizeIm))
                               
                        l += 1    

        #train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs_masks(image_label_pairs, train_images_folder, train_labels_folder, train_masks_folder)
        #save_pairs_masks(val_pairs, val_images_folder, val_labels_folder, val_masks_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l

def create_dataset_grain_OR(nbr_inputs, nbr_split, source_folder, image_folder, labels_folder, masks_folder, orientations_folder, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    gt_folder = os.path.join(source_folder, 'GB')
    
    data_files = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(data_files)
    gb = glob.glob(os.path.join(gt_folder, '*Twins_vf.png'))
    
    img = cv2.imread(data_files[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)
       
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X )
            sect_Y = int(n / crop_Y )
            sections.append([sect_X,sect_Y])
    
    grain_path = gb[0]
    grain_image = cv2.imread(grain_path)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_unknown = copy.deepcopy(grain_image)
    # Replace pixel values below 255 with 0
    
    
    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_twin[np.all(grain_image_twin == red, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == blue, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)

    #grain_couple[np.all(grain_couple == black, axis=-1)] = white
    image_label_pairs = []
    # Display all grain images

    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                im_input = np.zeros((crop[idx1][0], crop[idx1][1], nbr_inputs))
                idx2 = 0
                im_orientation = np.zeros((crop[idx1][0],crop[idx1][1],len(data_files)))
                for idx in range(len(data_files)):
                    img = cv2.imread(data_files[idx], cv2.IMREAD_GRAYSCALE)
                    if idx == 0 or idx == 4 or idx == 8:
                        im_input[:, :, idx2] = img[int(i * crop[idx1][0]):int((i + 1) * crop[idx1][0] ),
                                  int(j * crop[idx1][1] ):int((j + 1) * crop[idx1][1] )]
                        idx2 = idx2 + 1
                    im_orientation[:, :, idx] = img[int(i * crop[idx1][0] ):int((i + 1) * crop[idx1][0] ),
                              int(j * crop[idx1][1]):int((j + 1) * crop[idx1][1] )]
                

                final_grain_twin = grain_image_twin[int(i * crop[idx1][0] ):int((i + 1) * crop[idx1][0] ),
                                  int(j * crop[idx1][1] ):int((j + 1) * crop[idx1][1])]
               # final_grain_unknown = grain_image_unknown[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
               #                   int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                final_couple = grain_couple[int(i * crop[idx1][0] ):int((i + 1) * crop[idx1][0] ),
                                  int(j * crop[idx1][1]):int((j + 1) * crop[idx1][1] ),:]
                contour_points_grain_twin = find_contour_final(final_grain_twin,0)
       
                #contour_points_grain_unknown = find_contour_final_test(final_grain_unknown,im_input)
         
                [m1,n1] = np.shape(final_grain_twin)
                SizeIm = [n1,m1]
                
                output_filename = f'{l}_Input.png'
                txt_filename = f'{l}_Input.txt'
                foldername = f'{l}_Input'
                
                
                #if l == 37:               
                #    plot_bounding_boxes(im_input,contour_points_grain_twin,final_couple)
                
                image_label_pairs.append((im_input, im_orientation, final_grain_twin, contour_points_grain_twin, output_filename, txt_filename,foldername,SizeIm,len(data_files)))
                
                grain_image[grain_image < 255] = 0

                l += 1 
  

        save_pairs_masks_orientation(image_label_pairs, image_folder, labels_folder, masks_folder, orientations_folder)

        image_label_pairs = []

   
    return l

def save_pairs_masks_orientation(pairs, image_folder, labels_folder, masks_folder, orientations_folder):
    for im_input, im_orientation, final_grain_twin, contour_points_grain_twin, image_filename, txt_filename, foldername, sizeIm, nbr_im in pairs:
        folder_ori = os.path.join(orientations_folder, foldername)
        os.makedirs(folder_ori)
        for i in range(nbr_im):
            name = f'{i*10}.png'
            cv2.imwrite(os.path.join(folder_ori,name), im_orientation[:,:,i])
        cv2.imwrite(os.path.join(image_folder, image_filename), im_input)
        cv2.imwrite(os.path.join(masks_folder, image_filename), final_grain_twin)
        with open(os.path.join(labels_folder, txt_filename), 'w') as file:
            for contour_points in contour_points_grain_twin:
                file.write('1 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")


def save_pairs_masks(pairs, image_folder, labels_folder, masks_folder):
    for im_input, final_grain_twin, contour_points_grain_twin, image_filename, txt_filename, sizeIm in pairs:
        cv2.imwrite(os.path.join(image_folder, image_filename), im_input)
        cv2.imwrite(os.path.join(masks_folder, image_filename), final_grain_twin)
        with open(os.path.join(labels_folder, txt_filename), 'w') as file:
            for contour_points in contour_points_grain_twin:
                file.write('1 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")
 
                    

    
def save_pairs_tests(pairs, image_folder, labels_folder, grains_masks_folder, twins_masks_folder, both_masks_folder):
    for im_input, final_grain_grain, contour_points_grains, final_grain_twin, contour_points_grain_twin, image_filename, txt_filename, sizeIm in pairs:
        cv2.imwrite(os.path.join(image_folder, image_filename), im_input)
        cv2.imwrite(os.path.join(grains_masks_folder, image_filename), final_grain_grain)
        cv2.imwrite(os.path.join(twins_masks_folder, image_filename), final_grain_twin)
        
        rgb_image = cv2.cvtColor(final_grain_twin, cv2.COLOR_GRAY2BGR)
        grain_grain = cv2.cvtColor(final_grain_grain, cv2.COLOR_GRAY2BGR)
        # Set black pixels to red [B, G, R] = [0, 0, 255]
        rgb_image[final_grain_twin < 255] = [0, 0, 255]
        
        final_grain_twin = final_grain_twin.astype(np.uint8)

        # Scale to 0 and 255 if necessary
        if final_grain_twin.max() == 1:
            final_grain_twin = final_grain_twin * 255
        final_image = np.ones_like(grain_grain, dtype=np.uint8) * 255
        twin_mask = final_grain_twin < 255  # Region to color red
        black_mask = np.all(grain_grain == [0, 0, 0], axis=2) 
        # Apply red where twin_mask is True
        final_image[twin_mask] = [0, 0, 255]  # Red in BGR
        
        # Apply original black pixels
        final_image[black_mask] = [0, 0, 0]  # Keep black
        
        cv2.imwrite(os.path.join(both_masks_folder, image_filename), final_image)
        
        with open(os.path.join(labels_folder, txt_filename), 'w') as file:
            for contour_points in contour_points_grains:
                file.write('1 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/(sizeIm[0]-1)} {point[1]/(sizeIm[1]-1)}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")

            for contour_points in contour_points_grain_twin:
                file.write('2 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/(sizeIm[0]-1)} {point[1]/(sizeIm[1]-1)}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")

def create_dataset_test(nbr_inputs, nbr_split, source_folder, image_folder, labels_folder, l):
    
    data_folder = os.path.join(source_folder, 'Cropped')
    gt_folder = os.path.join(source_folder, 'GB')
    
    data_files = glob.glob(os.path.join(data_folder, '*.bmp'))
    img_total = len(data_files)
    gb = glob.glob(os.path.join(gt_folder, '*Twins_vf.png'))
    
    img = cv2.imread(data_files[0], cv2.IMREAD_GRAYSCALE)
    [m,n] = np.shape(img)
    
    train_images_folder = os.path.join(image_folder, 'train')
    val_images_folder = os.path.join(image_folder, 'val')
    train_labels_folder = os.path.join(labels_folder, 'train')
    val_labels_folder = os.path.join(labels_folder, 'val')
 #   test_image_folder = os.path.join(image_folder,'test')
#    test_labels_folder = os.path.join(labels_folder,'test')
    
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5: 
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X * 2 - 1)
            sect_Y = int(n / crop_Y * 2 - 1)
            sections.append([sect_X,sect_Y])
    
    grain_path = gb[0]
    grain_image = cv2.imread(grain_path)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    grain_image[grain_image < 255] = 0
    grain_couple = copy.deepcopy(grain_image)
    grain_image_twin = copy.deepcopy(grain_image)
    grain_image_unknown = copy.deepcopy(grain_image)
    # Replace pixel values below 255 with 0
    
    
    # Define the color for black and white in BGR format
    black = [0, 0, 0]
    white = [255, 255, 255]
    red = [255, 0, 0]
    blue = [0, 0, 255]
    
    grain_image_twin[np.all(grain_image_twin == black, axis=-1)] = white
    grain_image_twin[np.all(grain_image_twin == blue, axis=-1)] = white

    grain_image_twin = cv2.cvtColor(grain_image_twin, cv2.COLOR_RGB2GRAY)

    #grain_couple[np.all(grain_couple == black, axis=-1)] = white
    image_label_pairs = []
    # Display all grain images


    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                im_input = np.zeros((crop[idx1][0], crop[idx1][1], nbr_inputs))
                for idx in range(nbr_inputs):
                    if nbr_inputs > 3:
                        shift = idx
                    else:
                        if img_total > 3:
                            shift = idx * 4
                        else:
                            shift = idx
                    img = cv2.imread(data_files[shift], cv2.IMREAD_GRAYSCALE)
                    im_input[:, :, idx] = img[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                              int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                
            
                final_grain_twin = grain_image_twin[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
               # final_grain_unknown = grain_image_unknown[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
               #                   int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]
                final_couple = grain_couple[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2),:]
                contour_points_grain_twin = find_contour_final(final_grain_twin,1)
       
                #contour_points_grain_unknown = find_contour_final_test(final_grain_unknown,im_input)
         
                [m1,n1] = np.shape(final_grain_twin)
                SizeIm = [n1,m1]
                
                output_filename = f'{l}_Input.png'
                txt_filename = f'{l}_Input.txt'
                
               # if l == 37:               
               #     plot_bounding_boxes(im_input,contour_points_grain_twin,final_couple)
                
                image_label_pairs.append((im_input, contour_points_grain_twin, output_filename, txt_filename,SizeIm))
                
                grain_image[grain_image < 255] = 0

                l += 1 
                for m in range(3):
                    if m == 0:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_CLOCKWISE)
                       # final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_CLOCKWISE)
             
                    elif m == 1:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_90_COUNTERCLOCKWISE)
                     #   final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  
                    elif m == 2:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                        final_grain_rot_t = cv2.rotate(final_grain_twin, cv2.ROTATE_180)
                      #  final_grain_rot_u = cv2.rotate(final_grain_unknown, cv2.ROTATE_180)

                    contour_points_grain_twin = find_contour_final(final_grain_rot_t,1)
  
                   # contour_points_grain_unknown = find_contour_final_test(final_grain_rot_u,im_input)
              
                    output_filename_rot = f'{l}_Input.png'
                    txt_filename_rot = f'{l}_Input.txt'
                    [m1,n1] = np.shape(final_grain_rot_t)
                    SizeIm = [n1,m1]
                    image_label_pairs.append((im_input_rot, contour_points_grain_twin, output_filename_rot, txt_filename_rot,SizeIm))
                           
                    l += 1    

        train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs(train_pairs, train_images_folder, train_labels_folder)
        save_pairs(val_pairs, val_images_folder, val_labels_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l

def save_pairs(pairs, image_folder, labels_folder):
    for im_input, contour_points_grain_twin, image_filename, txt_filename, sizeIm in pairs:
        cv2.imwrite(os.path.join(image_folder, image_filename), im_input)
        with open(os.path.join(labels_folder, txt_filename), 'w') as file:
            for contour_points in contour_points_grain_twin:
                file.write('1 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]}' for point in contour_points))
                    file.write('\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")
'''   
            for contour_points in contour_points_grain_unknown:
                file.write('2 ')
                if isinstance(contour_points, tuple) and len(contour_points) == 4:
                    file.write(f'{contour_points[0]/sizeIm[0]} {contour_points[1]/sizeIm[1]} '
                               f'{contour_points[2]/sizeIm[0]} {contour_points[3]/sizeIm[1]}\n')
                elif isinstance(contour_points, list):
                    file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]} '
                                        f'{point[2]/sizeIm[0]} {point[3]/sizeIm[1]}' for point in contour_points) + '\n')
                else:
                    raise ValueError("Invalid contour points format. Expected a tuple or a list of tuples.")
'''
def save_pairs_manual(pairs, image_folder, labels_folder):
    for im_input, contour_points_grain_twin, image_filename, txt_filename, sizeIm in pairs:
        cv2.imwrite(os.path.join(image_folder, image_filename), im_input)
        with open(os.path.join(labels_folder, txt_filename), 'w') as file:
            for contour_points in contour_points_grain_twin:
                file.write('2 ')
                file.write(' '.join(f'{point[0]/sizeIm[0]} {point[1]/sizeIm[1]}' for point in contour_points))
                file.write('\n')

def create_dataset_model(nbr_split, image, gt,
                         train_images_folder, train_labels_folder,
                         val_images_folder, val_labels_folder, l):
    

    
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    [m,n,d] = np.shape(img)
    
    crop = []
    sections = []

    for i in range(1, nbr_split):
        if i == 4 or i == 5:  
            crop_X = int(m / i)
            crop_Y = int(n / i)
            crop.append([crop_X,crop_Y])
            sect_X = int(m / crop_X * 2 - 1)
            sect_Y = int(n / crop_Y * 2 - 1)
            sections.append([sect_X,sect_Y])

    
    grain_image = cv2.imread(gt)
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_BGR2RGB)
    
    # Replace pixel values below 255 with 0
    grain_image[grain_image < 255] = 0
    
    grain_image = cv2.cvtColor(grain_image, cv2.COLOR_RGB2GRAY)
    
    image_label_pairs = []
    train_pairs = []
    val_pairs = []
        
    for idx1 in range(len(sections)):   
        for i in range(sections[idx1][0]):
            for j in range(sections[idx1][1]):
                im_input = np.zeros((crop[idx1][0], crop[idx1][1], 3))
                im_input[:, :, :] = img[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                          int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2), :]

                    
                final_grain = grain_image[int(i * crop[idx1][0] / 2):int((i + 1) * crop[idx1][0] / 2 + crop[idx1][0] / 2),
                                  int(j * crop[idx1][1] / 2):int((j + 1) * crop[idx1][1] / 2 + crop[idx1][1] / 2)]

                contour_points_grain = find_contour_final(final_grain)
                [m1,n1] = np.shape(final_grain)
                SizeIm = [n1,m1]
                
                output_filename = f'{l}_Input.png'
                txt_filename = f'{l}_Input.txt'
                
                image_label_pairs.append((im_input, contour_points_grain, output_filename, txt_filename,SizeIm))
              
                l += 1 
                
                # Rotations
                for m in range(3):
                    if m == 0:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_CLOCKWISE)
                        final_grain_rot = cv2.rotate(final_grain, cv2.ROTATE_90_CLOCKWISE)
             
                    elif m == 1:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        final_grain_rot = cv2.rotate(final_grain, cv2.ROTATE_90_COUNTERCLOCKWISE)
                  
                    elif m == 2:
                        im_input_rot = cv2.rotate(im_input, cv2.ROTATE_180)
                        final_grain_rot = cv2.rotate(final_grain, cv2.ROTATE_180)

                    contour_points_grain = find_contour_final(final_grain_rot)

                    output_filename_rot = f'{l}_Input.png'
                    txt_filename_rot = f'{l}_Input.txt'
                    [m1,n1] = np.shape(final_grain_rot)
                    SizeIm = [n1,m1]
                    image_label_pairs.append((im_input_rot, contour_points_grain, output_filename_rot, txt_filename_rot,SizeIm))
                           
                    l += 1    

        train_pairs,val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=None,shuffle=False)
        save_pairs(train_pairs, train_images_folder, train_labels_folder)
        save_pairs(val_pairs, val_images_folder, val_labels_folder)   
        train_pairs = []
        val_pairs = []
        image_label_pairs = []

   
    return l







