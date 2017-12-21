import numpy as np
import keras

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input, Reshape, Permute, Average
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Lambda, Activation, GlobalAveragePooling2D, Conv2DTranspose, GlobalAveragePooling1D
from keras.layers import AtrousConvolution2D, Convolution2D
from keras.optimizers import Adam
from keras.models import load_model

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
IMG_SIZE = 400
PATCH_SIZE = 16
BATCH_SIZE = 32
NUM_EPOCHS = 1
TRAINING_SIZE = 100
TEST_SIZE = 50
a = 0.001

def dilnet1(input_width, input_height) -> Sequential:
    model = Sequential()
    model.add(Conv2D(16, 1, activation='relu', name='conv1_1', input_shape=(input_width, input_height, NUM_CHANNELS)))
    model.add(Conv2D(16, 1, activation='relu', name='conv1_2'))
    model.add(Conv2D(16, 1, activation='relu', name='conv2_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(32, 2, activation='relu', name='conv2_2'))
    model.add(Conv2D(32, 2, activation='relu', name='conv3_1'))

    model.add(AtrousConvolution2D(64, 3, atrous_rate=(2, 2), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 2, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1, 1, activation='linear', name='fc-final'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('sigmoid'))
    return model


def dilnet2(input_width, input_height) -> Sequential:
    model = Sequential()
    model.add(Conv2D(16, 2, activation='relu', input_shape=(input_width, input_height, NUM_CHANNELS)))
    model.add(Conv2D(16, 2, activation='relu'))
    model.add(Conv2D(16, 2, activation='relu'))
    
    model.add(Conv2D(32, 2, activation='relu'))
    model.add(Conv2D(32, 2, activation='relu'))
    model.add(Conv2D(32, 2, activation='relu'))

    model.add(AtrousConvolution2D(64, 4, atrous_rate=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1, 1, activation='linear'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('sigmoid'))
    return model
	
def FCN1(input_width, input_height):
	model = Sequential()
	model.add(Conv2D(32, 2, input_shape=(input_width, input_height, NUM_CHANNELS)))
	model.add(LeakyReLU(alpha=a))
	model.add(Conv2D(32, 2))
	model.add(LeakyReLU(alpha=a))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(32, 2))
	model.add(LeakyReLU(alpha=a))
	model.add(Conv2D(32, 2))
	model.add(LeakyReLU(alpha=a))
	model.add(Conv2D(32, 2))
	model.add(LeakyReLU(alpha=a))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, 2))
	model.add(LeakyReLU(alpha=a))
	model.add(Conv2D(64, 1))
	model.add(LeakyReLU(alpha=a))
	model.add(Dropout(0.25))

	model.add(Conv2DTranspose(1, kernel_size=15, padding='valid'))
	model.add(GlobalAveragePooling2D())
	model.add(Activation('sigmoid'))
	return model
	
	
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))
	
def get_pretrained_models():
	"""
    Load and return pretrained models
    """
	model1 = load_model('DilNet_16_enh.h5', custom_objects= {'f1': f1})
	model2 = load_model('FCN_16_enh2.h5', custom_objects= {'f1': f1})
	model3 = load_model('DilNet_16_enh2.h5', custom_objects= {'f1': f1})
	return model1, model2, model3
	
from data_handler import extract_data, extract_labels, extract_test_data, masks_to_submission, img_crop, label_to_img

def train_and_get_models(data_dir):	
	"""
    Train models on images from data_dir and return them
    """
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/' 
	
	train_data = extract_data(train_data_filename, TRAINING_SIZE)
	train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

	model1 = dilnet1(PATCH_SIZE, PATCH_SIZE)
	model2 = FCN1(PATCH_SIZE, PATCH_SIZE)
	model3 = dilnet2(PATCH_SIZE, PATCH_SIZE)
	
	for model in (model1, model2, model3):
		LR = 0.001
		DECAY = 0.00000
		adam = Adam(lr=LR, decay=DECAY)
		model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc',f1])
		model.fit(train_data, 1-train_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
	return model1, model2, model3

# Get prediction for given input image 
def keras_prediction_av3(model1, model2, model3, img):
    """
    Get combined predictions of an ensemble of three models for an image
    """
    data = np.asarray(img_crop(img, PATCH_SIZE, PATCH_SIZE))
    pred1 = 1-model1.predict(data)
    pred2 = 1-model2.predict(data)
    pred3 = 1-model3.predict(data)
    img_prediction = label_to_img(img.shape[0], img.shape[1], PATCH_SIZE, PATCH_SIZE, (pred1 + pred2 + pred3) / 3)

    return img_prediction
	
def get_predictions(model1, model2, model3, test_data_dir):
    """
    Get combined predictions of an ensemble of three models on test images in test_data_dir
    """
    test_data, file_names = extract_test_data(test_data_dir, TEST_SIZE)
    submission_filename = 'submission.csv'
    images = [keras_prediction_av3(model1, model2, model3, test_data[i]) for i in range(test_data.shape[0])]
    masks_to_submission(submission_filename, images, file_names)