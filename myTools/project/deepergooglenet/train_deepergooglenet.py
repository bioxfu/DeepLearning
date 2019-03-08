# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import tiny_imagenet_config as config
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import MeanPreprocessor
from utilities.callbacks import TrainingMonitor
from utilities.io import HDF5DatasetGenerator
from utilities.nn.cnn import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import argparse
import json
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
	height_shift_range=0.2, shear_range=0.15, zoom_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
bs = 64
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batchSize=bs, aug=aug,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batchSize=bs,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	opt = Adam(lr=1e-3)
	model = DeeperGoogLeNet.build(width=64, height=64, depth=3,
		classes=config.NUM_CLASSES, reg=0.0002)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])
# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}...".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}...".format(K.get_value(model.optimizer.lr)))

# construct the set of callbacks
fname = os.path.sep.join([args['checkpoints'], "weights.{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
monitor = TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH, start_at=args["start_epoch"])
callbacks = [checkpoint, monitor]

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // bs,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // bs,
	epochs=10,
	max_queue_size=bs * 2,
	callbacks=callbacks,
	verbose=1)

# close the HDF5 datasets
trainGen.close()
valGen.close()

