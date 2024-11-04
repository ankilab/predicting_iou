# Predicting semantic segmentation quality in laryngeal endoscopy images

In this repository, you will find the code associated with predicting the semantic segmentation quality in laryngeal endoscopy images.

## Requirements

We used Python 3.10 for our experiments. We heavily rely on the following libraries:
* numpy
* scipy
* TensorFlow (2.16.0) with Keras (3.1.1), for the best experience enable GPU support. 

## Inside this repository

### Creating the IRR dataset

We provide a Jupyter notebook (`Create IRR dataset.ipynb`) that takes the pre-processed BAGLS dataset ([you can download this here](https://zenodo.org/records/14034494)), creates a random, but seeded subset from the full dataset (here: 100 samples) and creates three shuffles for assessing inter- and intra-rater reliability.

### IoU training data generation

For creating artificial artifacts on the BAGLS training data, we use a multitude of different artifacts focusing on border-pixels, low scale noise and large scale patches using Perlin noise. You find the code in the Jupyter notebook `IoU training data generation.ipynb`.

### Training the classifier

We provide scripts for loading the training data for the IoU prediction deep neural network (`DataGenerator.py`) and training the actual model (`train.py`).

## How to cite

To be announced
