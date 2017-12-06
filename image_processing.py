#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def apply_mirror_boundary_conditions(coord, dim):
    """
    Return the correct coordinate according to mirror boundary conditions
        coord: a coordinate (x or y) in the image
        dim: the length of the axis of said coordinate
    """
    # If the coordinate is outside of the bounds of the axis, take its reflection inside the image
    if coord < 0:
        coord = -coord
    elif coord >= dim:
        coord =  2*(dim-1) - coord % (2*(dim-1))
    # Else, do nothing
    return coord


def getpixel(image, x, y):
    """
    Return the value of the pixel (x,y) in image, with mirror boundary condition
    Input:
        image: the image tensor; size: (width, height, channel_num)
        x: the x-coordinate of the pixel
        y: the y-ccordinate of the pixel
    Output:
        val: the value of the pixel; size: (channel_num,)
    """
    width = image.shape[0]
    height = image.shape[1]
    if x < 0:
        x = -x
    if x >= width:
        x = 2*width - 2 - (x % (2*width-2))
    if y < 0:
        y = -y
    if y >= height:
        y = 2*height - 2 - (y % (2*height-2))
    return image[x,y,:]


def get_window(image, window_size, centre_coordinates):
    """
    Get a window in image taking into account boundary conditions
        image: a numpy array representing our image
        window_size: an odd number specifying the size of the window
        centre_coordinates: a list containing the x-y coordinates of the window's central pixel
    """
    # Get convenient variables
    window_radius = (window_size - 1)/2
    i_centre, j_centre = (centre_coordinates[0], centre_coordinates[1])
    nrows, ncols = image.shape
    window = np.zeros((window_size, window_size))
    # Fill in the window array with pixels of the image
    for i in range(window_size):
        # Apply mirror boundary conditions on the x-coordinate
        i_mirrored = apply_mirror_boundary_conditions(i_centre + i - window_radius, nrows)
        for j in range(window_size):
            # Same for the y-coordinate
            j_mirrored = apply_mirror_boundary_conditions(j_centre + j - window_radius, ncols)
            # Fill in the window with the corresponding pixel
            window[i, j] = image[i_mirrored, j_mirrored]
    return window


def shift_to_the_right(image, window, centre_coordinates):
    nrows, ncols = image.shape
    window_size = len(window)
    window_radius = (window_size - 1)/2
    j_mirrored = apply_mirror_boundary_conditions(centre_coordinates[1] + 1 + window_radius, nrows)
    shifted = np.roll(window, -1, axis=0)
    for i in range(window_size):
        i_mirrored = apply_mirror_boundary_conditions(i_centre + i - window_radius, nrows)
        shifted[i, -1] = window[i_mirrored, j_mirrored]
    return shifted


def shift_to_the_bottom(image, window, centre_coordinates):
    nrows, ncols = image.shape
    window_size = len(window)
    window_radius = (window_size - 1)/2
    i_mirrored = apply_mirror_boundary_conditions(centre_coordinates[0] + 1 + window_radius, nrows)
    shifted = np.roll(window, -1, axis=1)
    for j in range(window_size):
        j_mirrored = apply_mirror_boundary_conditions(j_centre + j - window_radius, ncols)
        shifted[-1, j] = window[i_mirrored, j_mirrored]
    return shifted


def sliding_window(image, window_size):
    """
    Construct a list of sliding windows of given size on an image.
    The windows will slide from left to right and from up to down.
        image: a numpy array representing our image
        window_size: an odd number specifying the size of the window
    """
    nrows, ncols = image.shape
    windows = []
    for i in range(nrows):
        for j in range(ncols):
            # TODO: do not rebuild the entire window at each loop, some values do not move.
            window = get_window(image, window_size, [i, j])
