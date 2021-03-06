import config
from preprocessor import ImageToArrayPreprocessor
from preprocessor import SimplePreprocessor
from preprocessor import MeanPreprocessor
from preprocessor import CropPreprocessor
from dataIO import HDF5DatasetGenerator
from tools import rank_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json
import os

pre_train_model = config.TRAIN_MODELS[0]
image_size = config.IMAGES_SIZE
batch_size = config.BATCH_SIZE
output_path = config.OUTPUT_PATH
saved_model = os.path.sep.join([output_path, '{}_model.hdf5'.format(pre_train_model)])
model_accuracy = open(os.path.sep.join([output_path, '{}_model_accuracy'.format(pre_train_model)]), 'w')

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(image_size[pre_train_model], image_size[pre_train_model])
cp = CropPreprocessor(image_size[pre_train_model], image_size[pre_train_model])
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
itap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(saved_model)

# initialize the testing dataset generators, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
model_accuracy.write("[INFO] predicting on test data (no crops)...\n")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, batchSize=batch_size,
	preprocessors=[sp, mp, itap], classes=config.NUM_CLASSES)
predictions = model.predict_generator(testGen.generator(),
	steps=testGen.numImages // batch_size, max_queue_size=batch_size * 2)

# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
model_accuracy.write("[INFO] rank-1: {:.2f}%\n".format(rank1 * 100))
model_accuracy.write("[INFO] rank-5: {:.2f}%\n".format(rank5 * 100))
testGen.close()

# re-initialzie the testing set generator excluding the SimplePreprocessor
testGen = HDF5DatasetGenerator(config.TEST_HDF5, batchSize=batch_size,
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

# compute the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
model_accuracy.write("[INFO] predicting on test data (with crops)...\n")
(rank1, rank5) = rank_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
model_accuracy.write("[INFO] rank-1: {:.2f}%\n".format(rank1 * 100))
model_accuracy.write("[INFO] rank-5: {:.2f}%\n".format(rank5 * 100))
testGen.close()
model_accuracy.close()
