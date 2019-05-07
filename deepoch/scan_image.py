import matplotlib
matplotlib.use("Agg")
import config
from preprocessor import ImageToArrayPreprocessor
from preprocessor import AspectAwarePreprocessor
from preprocessor import MeanPreprocessor
from preprocessor import CropPreprocessor
from keras.models import load_model
import numpy as np
from PIL import Image
import sys
import os
import json
import cv2
import matplotlib.pyplot as plt
import progressbar

train_model = config.TRAIN_MODELS[0]
image_size = config.IMAGES_SIZE[train_model]
output_path = config.OUTPUT_PATH
saved_model = os.path.sep.join([output_path, '{}_model.hdf5'.format(train_model)])

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
itap = ImageToArrayPreprocessor()

model = None

def load_saved_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = load_model(saved_model)

def prepare_image(image):
	image = np.asarray(image)
	image = mp.preprocess(image)
	image = itap.preprocess(image)
	image = np.expand_dims(image, axis=0)
	return image

im = np.array(Image.open(sys.argv[1]), dtype=np.float32)
im = cv2.resize(im, (320, 320), interpolation=cv2.INTER_AREA)
im = im[:,:,::-1]

imgwidth = im.shape[1]
imgheight = im.shape[0]
step = 4
heatmapwidth = int((imgwidth - 32) / step)
heatmapheight = int((imgheight - 32) / step)

heatmap = np.zeros((heatmapheight, heatmapwidth), dtype=np.float32)

load_saved_model()

# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", 
			progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=heatmapheight, 
	widgets=widgets).start()

im2 = im

for (y,x), element in np.ndenumerate(heatmap):
	imgx = x * step + 16;
	imgy = y * step + 16;

	image = im[imgy-16:imgy+16,imgx-16:imgx+16,:].copy()
	image = prepare_image(image)
	y_prob = model.predict(image)
	y_prob = y_prob.mean(axis=0)
	pbar.update(y)
	#print(y_prob)
	if y_prob[1] > 0.9:
		im2 = cv2.rectangle(im2, pt1=(imgx-16, imgy-16), pt2=(imgx+16, imgy+16), color=(0,0,255), thickness=1)

pbar.finish()
cv2.imwrite(sys.argv[2], im2)

