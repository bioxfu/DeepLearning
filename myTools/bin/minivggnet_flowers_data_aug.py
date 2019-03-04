# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
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
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO]: Loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, itap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=32), 
	validation_data=(test_x, test_y), steps_per_epoch=len(train_x)//32, 
	epochs=100, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
