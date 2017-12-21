# README
Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT

## Libraires used: **TODO**
* `conda: ???`
* `python: ???`
* `numpy: ???`
* `scipy: ???`
   * If not already installed, instructions are [here](https://www.scipy.org/install.html)
* `scikit-learn: ???`
   * The installation instructions can be found [here](http://scikit-learn.org/stable/install.html)
* `keras: ???`
   * Keras requires either TensorFlow, Theano, or CNTK as backend engines. It is recommended to install [TensorFlow](https://www.tensorflow.org/install/) (you can install either CPU or GPU version, if you do not already have CUDA and cuDNN, CPU installation will be far easier)
   * The instructions to install Keras are [here](https://keras.io/#installation)
   * It also need [h5py](http://docs.h5py.org/en/latest/build.html) to load and save models to disk

### Hardware: **TODO**
* Could put here the hardware used to make the model in order to give an idea of the memory that we used, 
and for the time required to run the code as a function of CPU/GPU

## Launching `run.py` *TODO*
In order to use `run.py`, follow these steps:
1. Put the *training/* and *test_set_images/* folders at the same place as `run.py`
2. Open a terminal and run `python run.py`, you may want to add some flags at the end:
   * `--verbose`: allow the code to print the current state (e.g., current epoch, batch,...)
   * `--train_model`: train the model from scratch, and then predict the labels (if not set, the code will use a provided pre-trained model)
   * `--low_memory`: for computer with smaller memory, in case the variables don't fit (may slow down the code)
3. The training requires ~???min on our computers, without GPU acceleration (it requires around ??? GB of memory)  
   The prediction requires ~???min on our computers
4. A new file "*final_submission.csv*" has been created, which contains our prediction for the Kaggle test set

## Code architecture *TODO*
### `run.py` *TODO*
The main script to load the datasets, learn a model, predict the labels of the test set, and create a submission file.  
Only learn the model if the flag `--train_model` was used.

1 parameters can be changed here:
* the seed for the random number generation (default = 3)

### `data_handler.py`
Functions used by `run.py` to load the datasets, create labels out of the ground-truth images, and create a *.csv* submission file.

For the creation of the labels, a patch of 16-by-16 pixels is assigned a foreground label (= road) 
if strictly more than 25% of the pixels are roads in the ground-truth.

### `processing.py` *TODO*
Functions for processing the data.

Main functions:
* `extract_windows`: Extract the neighborhood of each patches of 16-by-16 pixel on the given array of images.
Each neighborhood (called `window` in the code) is a small images centered on the patch.  
*Note:* we have used mirror boundary conditions.

### `???.py`
???
