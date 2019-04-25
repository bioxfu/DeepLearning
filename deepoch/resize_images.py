from preprocessor import ImageToArrayPreprocessor
from preprocessor import SimplePreprocessor
from dataIO import SimpleImageLoader
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import progressbar

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-s", "--size", required=True, type=int,
                help="image size")
ap.add_argument("-o", "--output", required=True,
                help="path to output dataset")
args = vars(ap.parse_args())

os.mkdir(args["output"])

# Grab a random sample of images from the dataset
image_paths = np.array(list(paths.list_images(args["dataset"])))
#indexes = np.random.randint(0, len(image_paths), size=(1000,))
#image_paths = image_paths[indexes]
#print(image_paths)

# Initialize the image preprocessors
sp = SimplePreprocessor(args["size"], args["size"])
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleImageLoader(preprocessors=[sp, itap])

# initialize the progress bar
widgets = ["[INFO]: Resizing images.... ", progressbar.Percentage(), " ", 
			progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()


for (i, x) in enumerate(image_paths):
	(data, _, _) = sdl.load([x])
	fname=x.split('/')[-1]
	cv2.imwrite(os.path.sep.join([args["output"], fname]), data[0])
	pbar.update(i)

pbar.finish()
