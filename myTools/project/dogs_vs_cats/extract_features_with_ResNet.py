# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import AveragePooling2D
from sklearn.preprocessing import LabelEncoder
from utilities.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
ap.add_argument("-b", "--batch-size", type=int, default=8,
                help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a convinience variable
bs = args["batch_size"]

# grab the list of images that we'll be describing, then randomly 
# shuffle them to allow for easy trainning and testing splits via
# array slicing during training time 
print("[INFO]: Loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# extract the class label names from the image paths then encode the labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the ResNet50 network
# In keras 2.1.6 avg_pool is added to the network whether you include_top or not.
# But in keras 2.2.0; avg_pool layer is added only if you include_top
# https://github.com/keras-team/keras/issues/10469
print("[INFO] loading network...")
base_model = ResNet50(weights="imagenet", include_top=False)
x = base_model.output
x = AveragePooling2D((7, 7), name='avg_pool')(x)
model = Model(inputs=base_model.input, outputs=x)

# initialize the HDF5 dataset writer, then store the class label
# names in the dataset
# the final average pooling layer of ResNet is 2048-d
dataset = HDF5DatasetWriter((len(imagePaths), 2048),
	args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(imagePaths), bs):
	# extract the batch of images and labels, then initialize the 
	# list of actual images that will be passed through the network
	# for feature extraction

	batchPaths = imagePaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	# loop over the images and labels in the current batch
	for (j, imagePath) in enumerate(batchPaths):
		# load the input image using the Keras helper utility
		# while ensuring the image is resized to 224*224 pixles
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)


		# preprocess the image by (1) expanding the dimensions and
		# (2) subtracting the mean RGB pixel intensity from the 
		# ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

		# add the image to the batch
		batchImages.append(image)

	# pass the images through the network and use the outputs as
	# out actual features
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)

	# reshape the features so that each image is represented by 
	# a flattened feature vector of the 'MaxPooling2D' output
	features = features.reshape((features.shape[0], 2048))

	# add the features and labels to our HDF5 dataset
	dataset.add(features, batchLabels)
	pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()

