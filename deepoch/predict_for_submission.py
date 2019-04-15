import config
from preprocessor import ImageToArrayPreprocessor
from preprocessor import MeanPreprocessor
from preprocessor import CropPreprocessor
from dataIO import HDF5DatasetGenerator
from keras.models import load_model
import numpy as np
import progressbar
import json
import os

pre_train_model = config.PRE_TRAIN_MODELS[0]
image_size = config.IMAGES_SIZE
batch_size = config.BATCH_SIZE
output_path = config.OUTPUT_PATH
saved_model = os.path.sep.join([output_path, '{}_model.hdf5'.format(pre_train_model)])

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
cp = CropPreprocessor(image_size[pre_train_model], image_size[pre_train_model])
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
itap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(saved_model)

print("[INFO] predicting on test data (with crops)...")
testGen = HDF5DatasetGenerator(config.PREDICT_HDF5, batchSize=batch_size,
	preprocessors=[mp], classes=config.NUM_CLASSES)
predictions = []

# initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ", 
			progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // batch_size, 
	widgets=widgets).start()

# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
	# loop over each of the individual images
	for image in images:
		# apply the crop preprocessor to the image to generate 10
		# separate crops, then convert them from images to arrays
		crops = cp.preprocess(image)
		crops = np.array([itap.preprocess(c) for c in crops],
			dtype="float32")

		# make predictions on the crops and then average them
		# together to obtain the final predictions
		pred = model.predict(crops)
		predictions.append(pred.mean(axis=0))

	# update the progress bar
	pbar.update(i)

testGen.close()

output = open(config.PREDICT_CSV, 'w')
output.write('id,label\n')

for (i, p) in enumerate(predictions):
	#probability that the image is a dog (1 = dog, 0 = cat).
	output.write('{},{}\n'.format(i+1, p[1]))
output.close()
