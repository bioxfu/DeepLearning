# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.callbacks import TrainingMonitor
from utilities.nn.cnn import LeNet, MiniGoogLeNet, MiniVGGNet, ResNet
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
ap.add_argument("-w", "--weights", required=True,
                help="path to the best model weights file")
ap.add_argument("-e", "--epochs", required=True, type=int,
                help="number of epoches")
args = vars(ap.parse_args())

# define the total number of epochs to train for along with the 
# initial learning rate
NUM_EPOCHS = args["epochs"]
INIT_LR = 5e-3

def step_decay(epoch):
    # Initialize the base initial learning rate, drop factor, and epochs to drop every
    init_alpha = 0.01
    factor = 0.5
    drop_every = 5
    # Compute learning rate for the current epoch
    alpha = init_alpha * (factor ** np.floor((1 + epoch) / drop_every))
    # Return the learning rate
    return float(alpha)

def poly_decay(epoch):
    # Initialize the maximum number of epochs, base learning rate
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # Compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # Return the learning rate
    return alpha


# Show information on the process ID
print("[INFO]: Process ID: {}".format(os.getpid()))

# Load the training and testing data, then scale it into the range [0, 1]
print('[INFO]: Loading CIFAR-10 data....')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
#trainX = trainX.astype("float") / 255.0
#testX = testX.astype("float") / 255.0
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

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
#optimizer = SGD(lr=0.01, decay=0.01/args["epochs"], momentum=0.9, nesterov=True)
optimizer = SGD(lr=INIT_LR, momentum=0.9, nesterov=True)
#model = LeNet.build(width=32, height=32, depth=3, classes=10)
#model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
#model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath)]

# Construct the callback to save only the 'best' model to disk based on the validation loss
checkpoint = ModelCheckpoint(args['weights'], monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks.append(checkpoint)

# Learning rate decay for (MiniGoogLeNet)
callbacks.append(LearningRateScheduler(poly_decay))
#callbacks.append(LearningRateScheduler(step_decay))

# Train the network
print("[INFO]: Training....")
#H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=args["epochs"], callbacks=callbacks, verbose=1)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), 
	validation_data=(testX, testY), steps_per_epoch=len(trainX)//32, 
	epochs=args["epochs"], callbacks=callbacks, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

