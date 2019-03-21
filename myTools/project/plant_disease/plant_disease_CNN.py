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
from sklearn.model_selection import train_test_split
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
ap.add_argument("-w", "--weights", required=True,
                help="path to the best model weights file")
ap.add_argument("-e", "--epochs", required=True, type=int,
                help="number of epoches")
ap.add_argument("-s", "--image-size", required=True, type=int,
                help="size of images")
ap.add_argument("-b", "--batch-size", required=True, type=int,
                help="batch size of images to be passed through network")
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

# Grab the list of images
print("[INFO]: Loading images....")
image_paths = list(paths.list_images(args["dataset"]))

#print(image_paths)

# Initialize the image preprocessors
img_size = args["image_size"]
sp = SimplePreprocessor(img_size, img_size)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

# Split the data into training data (75%) and testing data (25%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# Initialize the label names 
labelNames = lb.classes_

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.01, decay=0.01/args["epochs"], momentum=0.9, nesterov=True)
#optimizer = SGD(lr=INIT_LR, momentum=0.9, nesterov=True)
#model = LeNet.build(width=img_size, height=img_size, depth=3, classes=len(labelNames))
#model = MiniVGGNet.build(width=img_size, height=img_size, depth=3, classes=len(labelNames))
#model = MiniGoogLeNet.build(width=img_size, height=img_size, depth=3, classes=len(labelNames))
model = ResNet.build(img_size, img_size, 3, len(labelNames), (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
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
batch_size = args["batch_size"]
print("[INFO]: Training....")
#H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=args["epochs"], callbacks=callbacks, verbose=1)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size), 
	validation_data=(testX, testY), steps_per_epoch=len(trainX)//batch_size, 
	epochs=args["epochs"], callbacks=callbacks, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames), digits=4)

