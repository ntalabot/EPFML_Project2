# -*- coding: utf-8 -*-
"""
Module containing functions to handle the data, as loading, creating labels,
and creating a submission file

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""

import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

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

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx] > 0.5:
                array_labels[j:j+w, i:i+h] = 1
            else:
                array_labels[j:j+w, i:i+h] = 0
            idx = idx + 1
    return array_labels
	
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches
			
			
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    tmp_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            tmp_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(tmp_imgs)
    for i in range(num_images):
        gt_imgs.append(np.asarray(tmp_imgs[i]) / 255)        
    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(num_images)]
    data = np.array([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.array([value_to_class(np.mean(data[i])) for i in range(len(data))])

    return labels.astype(np.float32)

def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
	
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    tmp_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            tmp_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(tmp_imgs)
    for i in range(num_images):
        imgs.append(np.asarray(tmp_imgs[i]) / 255)
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/PATCH_SIZE)*(IMG_HEIGHT/PATCH_SIZE)
    print('Patches per images: ' + str(N_PATCHES_PER_IMAGE))

    img_patches = [img_crop(imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(num_images)]

    data = np.array([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    return data
	
def extract_test_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    names = []
    for i in range(1, num_images+1):
        imageid = "test_{}/test_{}".format(i,i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
            names.append(image_filename)
        else:
            print ('File ' + image_filename + ' does not exist')
    return np.array(imgs), names
	
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

import re
def mask_to_submission_strings(im, im_name):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", im_name).group(0))
    for j in range(0, im.shape[1], PATCH_SIZE):
        for i in range(0, im.shape[0], PATCH_SIZE):
            patch = im[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, images, images_names):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for ind, fn in enumerate(images):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, images_names[ind]))