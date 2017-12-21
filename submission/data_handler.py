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
		
def value_to_class(v):
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
    df = np.sum(v)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):    
    """
    Reconstruct predicted segmentation from labels array
    """
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
    """
	Cut input image into pieces of size w x h and return list of them 
	"""
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
    """
	Extract the labels as array of integers from set {0,1}
	"""
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


	
def extract_data(filename, num_images):
    """
	Extract train images into a 4D tensor [image index, y, x, channels].
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
    """
	Extract test images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    names = []
    for i in range(1, num_images+1):
        imageid = "test_{}/test_{}".format(i,i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)
            names.append(image_filename)
        else:
            print ('File ' + image_filename + ' does not exist')
    return np.array(imgs), names
	

import re
def mask_to_submission_strings(im, im_name):
    """
    Reads a single image and outputs the strings that should go into the submission file
    Input:
        im (string): image labels
        im_name (string): image file name
    """
    img_number = int(re.search(r"\d+", im_name).group(0))
    for j in range(0, im.shape[1], PATCH_SIZE):
        for i in range(0, im.shape[0], PATCH_SIZE):
            patch = im[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, images, images_names):
    """
    Create a submssion file (.csv) from the given label images
    Input:
        submission_filename (string): the name to be given to the submission file
        images (string): the labels array; size:(num_image, image_size/PATCH_SIZE, image_size/PATCH_SIZE)
        images_names: file names of all output images
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for ind, fn in enumerate(images):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, images_names[ind]))