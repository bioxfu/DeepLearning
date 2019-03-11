# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from utilities.nn.cnn import ResNet
from utilities.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import argparse
import sys
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
ap.add_argument("-r", "--learning-rate", type=float, required=True,
                help="learning rate of optimizer")
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

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	opt = SGD(lr=args['learning_rate'])
	model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
		(64, 64, 128, 256), reg=0.0005)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}...".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, args['learning_rate'])
	print("[INFO] new learning rate: {}...".format(K.get_value(model.optimizer.lr)))

# construct the set of callbacks
os.mkdir(os.path.sep.join([args['checkpoints'], "lr_"+str(args['learning_rate'])]))
fname = os.path.sep.join([args['checkpoints'], "lr_"+str(args['learning_rate']), "{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
monitor = TrainingMonitor("output/resnet56_cifar10.png", json_path="output/resnet56_cifar10.json", start_at=args["start_epoch"])
callbacks = [checkpoint, monitor]

# train the network
bs = 128
model.fit_generator(
	aug.flow(trainX, trainY, batch_size = bs),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // bs,
	epochs=100,
	callbacks=callbacks,
	verbose=1)
