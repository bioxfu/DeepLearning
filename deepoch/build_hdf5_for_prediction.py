import config
from preprocessor import AspectAwarePreprocessor
from dataIO import HDF5DatasetWriter
from imutils import paths
import progressbar
import cv2

IMAGES_SIZE = 300

# grab the paths to the images
predictPaths = list(paths.list_images(config.PREDICT_PATH))
predictLabels = [0 for p in predictPaths]

# initialize the image preprocessor and the lists of RGB channel
# average
aap = AspectAwarePreprocessor(IMAGES_SIZE, IMAGES_SIZE)

# create HDF5 writer
print("[INFO] building {}...".format(config.PREDICT_HDF5))
writer = HDF5DatasetWriter((len(predictPaths), IMAGES_SIZE, 
	IMAGES_SIZE, 3), config.PREDICT_HDF5)

# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", 
			progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(predictPaths), 
	widgets=widgets).start()

# loop over the image paths
for (i, (path, label)) in enumerate(zip(predictPaths, predictLabels)):
	# load the image and process it
	image = cv2.imread(path)
	image = aap.preprocess(image)

	# add the image to the HDF5 dataset
	writer.add([image], [label])
	pbar.update(i)

# close the HDF5 writer
pbar.finish()
writer.close()
