# README
Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT

## Libraires used:
* `conda: 4.3.30`
* `python: 3.6.2`
* `numpy: 1.13.3`
* `scipy: 0.19.1`
* `h5py: 2.7.0`
   * If not already installed, instructions are [here](https://www.scipy.org/install.html)
* `scikit-learn: 0.19.1`
   * The installation instructions can be found [here](http://scikit-learn.org/stable/install.html)
* `keras: 2.0.4 (TensorFlow 1.3.0)`
   * Keras requires either TensorFlow, Theano, or CNTK as backend engines. It is recommended to install [TensorFlow](https://www.tensorflow.org/install/) (you can install either CPU or GPU version, if you do not already have CUDA and cuDNN, CPU installation will be far easier)
   * The instructions to install Keras are [here](https://keras.io/#installation)
   * It also need [h5py](http://docs.h5py.org/en/latest/build.html) to load pretrained models to disk

## Launching `run.py`
In order to use `run.py`, follow these steps:
1. Put the *training/* and *test_set_images/* folders at the same place as `run.py`
2. Put the pretrained model files (*.h5) at the same place as `run.py`
3. Open a terminal and run `python run.py`, you may want to add some flags at the end:
   * `--train_model`: train the model from scratch, and then predict the labels (if not set, the code will use a provided pre-trained model)
3. The training requires ~3min/epoch on our computers, without GPU acceleration
   The prediction requires ~5min on our computers
4. A new file "submission.csv*" has been created, which contains our prediction for the Kaggle test set

## Code architecture 
### `run.py`
The main script to load the datasets, learn a model, predict the labels of the test set, and create a submission file.
Parameters to change: directories of train and test images 
Only learn the model if the flag `--train_model` was used.

### `data_handler.py`
Functions used by `run.py` to load the datasets, create labels out of the ground-truth images, and create a *.csv* submission file.

### `models.py`
Model descriptions in Keras, functions to train them/load pretrained and make predictions

For the creation of the labels, a patch of 16-by-16 pixels is assigned a foreground label (= road) 
if strictly more than 25% of the pixels are roads in the ground-truth.