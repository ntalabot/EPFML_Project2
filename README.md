# EPFML_Project2
Road segmentation project for the Machine Learning Course at EPFL

## Ideas:
* As the dataset is imbalanced, we should do something:
   * e.g.: with Binary Cross Entropy, give weights to each class to increase the loss of the smallest class (here road), and decrease the loss of the biggest. (I have used ratio = road / (raod+background) to compute the weights: [ratio, 1-ratio])

* Try to add an edge detection before the network, and see if results improve

* Can use `del x` in order to delete a variable and free some memory space during the execution of the program

## TODO:
0. Choose librairies to work with for the final submission (maybe we can try out models and cross-validation with multiple I guess, but we should stick with one for the `run.py`, and for the report)

1. *Rigoroulsy* find a good model:
   1. Compare to some baseline methods (taken from the project description):
   
      "[...] include several properly implemented baseline algorithms as a comparison to your approach."
   
   2. What we have tried, different "steps", and if they did work, or not
      * compare the evolution of evaluation metric, i.e. F1-score

2. Report (4 pages PDF)

   **OverLeaf link:** https://www.overleaf.com/12880986jnctkgpxbpwq#/49224066/
   
   Explain our steps to find our final model, etc.

3. Code:
   1. `run.py`: to reproduce our final submission *.csv* file
   2. with its README (explain how to run the code, install librairies, etc.)
