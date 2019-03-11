# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from utilities.nn.cnn import ResNet
from utilities.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import sys
import os

# define the total number of epochs to train for along with the 
# initial learning rate
NUM_EPOCHS = 100
INIT_LR = 1e-1

def poly_decay(epoch):
    # Initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # Compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # Return the learning rate
    return alpha

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# Load the training and testing data, converting the images from
# integers to floats
print("[INFO]: Loading CIFAR-10 data....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
	(64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the set of callbacks
fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(fig_path, json_path), LearningRateScheduler(poly_decay)]

# train the network
bs = 128
model.fit_generator(
	aug.flow(trainX, trainY, batch_size = bs),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // bs,
	epochs=100,
	callbacks=callbacks,
	verbose=1)

# save the network do disk
print("[INFO] serializing network...")
model.save(args["model"])
