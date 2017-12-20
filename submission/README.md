# README 
# *TODO - For the moment most things are taken from the previous project*
Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT

Version used: **TODO**
* `conda: ???`
* `python: ???`
* `numpy: ???`
* `???: ???`

Hardware: **TODO**
* Could put here the hardware used to make the model in order to give an idea of the memory that we used, 
and for the time required to run the code as a function of CPU/GPU

### How to install *TODO*
Put here step to install additional libraires

## Launching `run.py` *TODO - Add flags explication, etc. (e.g. CPU/GPU, use pre-trained model, verbose, ...)*
In order to use `run.py`, follow these steps:
1. Put the *training/* and *test_set_images/* folders at the same place as `run.py`
2. Open a terminal and run `python run.py`
3. Wait for it to compute the model and create the submission file (it should take 3 to 5 min)
4. A new file *run_submission.csv* has been created, which contains the prediction for the Kaggle competition

## Code architecture *TODO*
### `run.py` *TODO*
The script to load the dataset, learn a model, predict the labels of the test set, and create a submission file.

3 parameters can be changed here:
* the seed for the random number generation (default = 3)
* the degree of the polynomial features (default = 11)
* the ratio of the train set actually going to training (default = 0.66)

### `data_handler.py`
Functions used by `run.py` to load the datasets, create labels out of the ground-truth images, and create a *.csv* submission file.

For the creation of the labels, a patch of 16-by-16 pixels is assigned a foreground label (= road) 
if more than 25% of the pixels are roads in the ground-truth.

### `processing.py` *TODO*
Functions for processing the data.

Main functions:
* `extract_windows`: Extract the neighborhood of each patches of 16-by-16 pixel on the given array of images.
Each neighborhood (called `window` in the code) is a small images centered on the patch.  
*Note:* we have used mirror boundary conditions.

### `???.py`
???
