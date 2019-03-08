# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
                help="path to models directory")
args = vars(ap.parse_args())

# Load the testing data, then scale it into the range [0, 1]
(test_x, test_y) = cifar10.load_data()[1]
test_x = test_x.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)

# Initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# construct the path used to collect the models then initialize the 
# models list
model_paths = os.path.sep.join([args["models"], "*.model"])
model_paths = list(glob.glob(model_paths))
models = []

# loop over the model paths, loading the model, and adding it to 
# the list of models
for (i, model_path) in enumerate(model_paths):
	print("[INFO] loading model {}/{}".format(i+1, len(model_paths)))
	models.append(load_model(model_path))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions = []

# loop over the models
for model in models:
	# use the current model to make predictions on the testing data,
	# then store these predictions in the aggregate predictions list
	predictions.append(model.predict(test_x, batch_size=64))

# average the probabilities across all model predictions, then show
# a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

