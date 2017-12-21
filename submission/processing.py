# -*- coding: utf-8 -*-
"""
Module containing functions to process the data, as window extraction, ???

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""

import numpy as np

from data_handler import PATCH_SIZE

def get_pixel(image, i, j):
    """
    Return the value of the pixel (i,j) in image, with mirror boundary conditions
    Note that (i,j) = (y,x) in pixel coordinate (assuming image conventions)
    Input:
        image (numpy.ndarray): the image array
                               size:(width, height, num_channel) OR (width, height)
        i: the row-coordinate of the pixel
        j: the col-ccordinate of the pixel
    Output:
        the value(s) of the pixel; size:(num_channel,) OR simple scalar
    """
    width = image.shape[0]
    height = image.shape[1]
    if i < 0:
        i = -i
    if i >= height:
        i = 2*height - 2 - (i % (2*height-2))
    if j < 0:
        j = -j
    if j >= width:
        j = 2*width - 2 - (j % (2*width-2))
    if len(image.shape) == 3:
        return image[int(i),int(j),:]
    else:
        return image[int(i),int(j)]


def get_window(image, window_size, i, j):
    """
    Extract a square window in 'image', with the top left corner at (i,j) (mirror boundary conditions apply)
    Note that (i,j) = (y,x) in pixel coordinate (assuming image conventions)
    Input:
        image (numpy.ndarray): the image array
                               size:(height, width, num_channel)
        window_size (int): size of the window's side to extract
        i: the row-coordinate of the window's top-left corner
        j: the col-coordinate of the window's top-left corner
    Output:
        window (numpy.ndarray): the window array
                                size:(window_size, window_size, num_channel)
    """
    window = np.zeros((window_size, window_size, image.shape[-1]))
    for ii in range(window_size):
        for jj in range(window_size):
            window[ii,jj,:] = get_pixel(image, i + ii, j + jj)
    return window


def shift_window(image, window, i, j, stride=1):
    """
    Shift the window to the right by 'stride' pixels on the image (mirror boundary conditions apply)
    Note that (i,j) = (y,x) in pixel coordinate (assuming image conventions)
    Input:
        image (numpy.ndarray): the image array
                               size:(height, width, num_channel)
        window (numpy.ndarray): the window array
                                size:(window_size, window_size, num_channel)
        i: the row-coordinate of the window's top-left corner
        j: the col-coordinate of the window's top-left corner
        stride: the number of pixel to move the window (default: 1)
    Output:
        new_window (numpy.ndarray): the shifted window
                                    size:(window_size, window_size, num_channel)
    """
    window_size = window.shape[0]
    new_window = np.zeros(window.shape)
    new_i = i
    new_j = j + stride

    # If stride is too big, so no overlap between windows, return a new window
    if stride >= window_size:
        return get_window(image, window_size, new_i, new_j)
    
    # Else, delete first 'stride' columns, and insert 'stride' new ones at the end of the window
    new_window[:,:window_size-stride,:] = window[:,stride:,:]
    for ii in range(window_size):
        for jj in range(window_size-stride, window_size):
            new_window[ii,jj,:] = get_pixel(image, new_i + ii, new_j + jj)
    return new_window
    

def image_to_windows(image, window_size=PATCH_SIZE):
    """
    Return the windows around each patch of pixels of the given image (mirror boundary conditions apply)
    Input:
        image (numpy.ndarray): the image array
                               size:(height, width, num_channel)
        window_size (int): size of the windows' side
                           should be even, and bigger than PATCH_SIZE
                           (default: PATCH_SIZE)
    Output:
        windows (numpy.ndarray): the windows array
                                 size:((height*width/PATCH_SIZE)**2, 
                                       window_size, window_size, num_channel)
    """
    height, width, _ = image.shape
    # Offset between patch's and window's top-left corners
    offset = -(window_size - PATCH_SIZE)//2
    windows_list = []
    
    # Get first window of the row, then loop on the columns
    for i in range(offset, height+offset, PATCH_SIZE):
        window = get_window(image, window_size, i, offset)
        windows_list.append(window)
        for j in range(offset, width-PATCH_SIZE+offset, PATCH_SIZE):
            window = shift_window(image, window, i, j, stride=PATCH_SIZE)
            windows_list.append(window)
    
    windows = np.asarray(windows_list)
    return windows


def extract_windows(images, window_size=PATCH_SIZE, verbose=False):
    """
    Return the windows around each patch of pixels in every images (mirror boundary conditions apply)
    Input:
        image (numpy.ndarray): the images array
                               size:(num_images, height, width, num_channel)
        window_size (int): size of the windows' side
                           should be even, and bigger than PATCH_SIZE
                           (default: PATCH_SIZE)
        verbose (boolean): flag to print the current state (default: False)
    Output:
        windows (numpy.ndarray): the windows array
                                 size:(num_images, (height*width/PATCH_SIZE)**2, 
                                       window_size, window_size, num_channel)
    """
    windows = []
    if verbose:
        print("Extracting the windows:\n")
    for i in range(0, images.shape[0]):
        if verbose:
            print("Extracting from image [%d/%d]" % (i+1, images.shape[0]))
        windows.append(image_to_windows(images[i], window_size=window_size))
    return np.asarray(windows)


