# -*- coding: utf-8 -*-
"""
Module containing functions to handle the data, as loading, creating labels,
and creating a submission file

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""

import os
import matplotlib.image as mpimg
import numpy as np

PATCH_SIZE = 16
FOREGROUND_THRESHOLD = 0.25

def load_images(dir_name, subfolder=False):
    """
    Load images from a directory, and return them as a numpy array
    Input:
        dir_name (string): the name of the directory containing the images
                           automatically adds '/' at the end if necessary 
        subfolder (boolean): flag indicating if the images are split
                             in multiple subfolders (assumes 1 image per folder)
    Output:
        data (numpy.ndarray): the array with the loaded images
                              the images are loaded in alphabetical order
                              size:(num_images, height, width, num_channel)
                              if num_channel = 1, this dimension is discarded
    """
    if dir_name[-1] != '/':
        dir_name += '/'
    
    # Extract file names (with subfolder's name if applicable)
    if subfolder:
        filenames = []
        subfolder_names = sorted(os.listdir(dir_name))
        for i in range(subfolder_names):
            filename = os.listdir(dir_name)[0]
            filenames.append(subfolder_names[i] + '/' + filename)
    else:
        filenames = sorted(os.listdir(dir_name))
    
    # Load the images
    data = []
    for i in range(len(filenames)):
        image = mpimg.imread(dir_name + filenames[i])
        data.append(image)
        
    data = np.asarray(data)
    return data


def patch_to_label(patch):
    """
    Return the label of the pixel patch, by comparing the ratio of foreground 
    to FOREGROUND_THRESHOLD
    Input:
        patch (numpy.ndarray): patch of a groundtruth image
                               size:(PATCH_SIZE, PATCH_SIZE)
    Output:
        the label of the patch: 
            1: foreground
            0: background
    """
    fg = np.mean(patch)
    if fg > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def gt_to_label(gt_image):
    """
    Return a label image, based on the given ground-truth image
    Input:
        gt_image (numpy.ndarray): the groundtruth image
                                  size:(height, width)
    Output:
        label (numpy.ndarray): the label image
                               size:(height/PATCH_SIZE, width/PATCH_SIZE)
    """
    height, width = gt_image.shape
    label = np.zeros((height//PATCH_SIZE, width//PATCH_SIZE))
    # For each patch: extract and label it
    for i in range(0, height//PATCH_SIZE):
        for j in range(0, width//PATCH_SIZE):
            patch = gt_image[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE]
            label[i,j] = patch_to_label(patch)
    return label


def create_labels(gt_images):
    """
    Return a label image, based on the given ground-truth image
    Input:
        gt_images (numpy.ndarray): the groundtruth images
                                  size:(num_images, height, width)
    Output:
        labels (numpy.ndarray): the label images
                               size:(num_images, height/PATCH_SIZE, width/PATCH_SIZE)
    """
    labels = []
    for i in range(gt_images.shape[0]):
        labels.append(gt_to_label(gt_images[i]))
    return np.asarray(labels)


def label_to_submission(img_num, label):
    """
    Generate strings from the label image, formated for the submission
    Input:
        img_num (int): the number of the image corresponding to the label array
        label (numpy.ndarray): the label array
                               size:(image_size/PATCH_SIZE, image_size/PATCH_SIZE)
    Yield:
        a string to be written in the submission file corresponding to a patch labeling
    """
    for j in range(label.shape[0]):
        for i in range(label.shape[1]):
            yield "{:03d}_{}_{},{}".format(img_num, j*PATCH_SIZE, i*PATCH_SIZE, label[i][j])


def create_submission(filename, labels):
    """
    Create a submssion file (.csv) from the given label images
    Input:
        filename (string): the name to be given to the submission file
                           automatically adds ".csv" at the end if necessary 
        labels: the labels array; size:(num_image, image_size/PATCH_SIZE, image_size/PATCH_SIZE)
    """
    if filename[-4:] != ".csv":
        filename += ".csv"
    
    with open(filename, 'w') as f:
        f.write('id,prediction\n')
        # Loop on the images to write in the submission file
        for i in range(labels.shape[0]):
            f.writelines('{}\n'.format(s) for s in label_to_submission(i+1, labels[i]))


