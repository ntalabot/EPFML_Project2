# -*- coding: utf-8 -*-
"""
Main file for the second Machine Learning project:
    ### TODO ###

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""

import argparse

from data_handler import load_images, create_labels, create_submission
from processing import extract_windows

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="allow printing in the code (e.g. the current Epoch,...)",
                    action="store_true")
parser.add_argument("--pre_trained", help="use the existing model to make the prediction, without training a new one",
                    action="store_true")
parser.add_argument("--low_memory", help="for computer with lower memory, may slow down the code",
                    action="store_true")
args = parser.parse_args()

# Load datasets, and create labels
if args.verbose:
    print("Loading the datasets")
x_train = load_images("training/images/")
gt_train = load_images("training/groundtruth/")
y_train = create_labels(gt_train)
x_test = load_images("test_set_images/", subfolder=True)

# Constants and Parameters
### TODO ###
seed = 3
window_size = 32

# Learn the model
### TODO ###
## /!\ Might be too big for memory, maybe create a flag to extract windows 
## batch per batch for computer with lower Memory
## Or even just do it in the batches ? But harder&slower!
tx = extract_windows(x_train, window_size=window_size, verbose=args.verbose)
ty = y_train.reshape((y_train.shape[0], -1))

# Predict the test labels, and create a Kaggle submission
### TODO ###
## /!\ Extracting window for all images is bigger than for training
## Thus memory problem! => will have to loop over images, but that's okay
x_kaggle = 0
labels_pred = 0
create_submission("final_submission", labels_pred)
