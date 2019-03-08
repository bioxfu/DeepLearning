# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-b", "--batch-size", type=int, default=8,
                help="batch size of images to be passed through network")
args = vars(ap.parse_args())

# store the batch size in a convinience variable
bs = args["batch_size"]

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO]: Loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
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
test_y = lb.fit_transform(test_y)

# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# initialilze the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, output=headModel)

# loop over all layers in the base model and freeze them so they
# will not be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO]: Compiling model....")
optimizer = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train the head of the network for a few epoches (all other 
# laysers are frozen) -- this will allow the new FC layers to 
# start to become initialize with actual "learned" values
# versus pure random 
print("[INFO]: Training head....")
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=bs), 
	validation_data=(test_x, test_y), steps_per_epoch=len(train_x)//32, 
	epochs=25, verbose=1)

# evaluate the network after initialization
print("[INFO]: Evaluating after initialization....")
predictions = model.predict(test_x, batch_size=bs)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
	layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a very small learning rate
print("[INFO]: Re-compiling model....")
optimizer = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train teh model again, this time fine-tuning both the final set
# of CONV layers along with our set of FC layers
print("[INFO]: Fine-tuning model....")
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=bs), 
	validation_data=(test_x, test_y), steps_per_epoch=len(train_x)//32, 
	epochs=100, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO]: Evaluating after fine-tuning....")
predictions = model.predict(test_x, batch_size=bs)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# save the model to disk
print("[INFO]: Serializing model....")
model.save(args["model"])

