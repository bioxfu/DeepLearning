# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.nn.cnn import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
ap.add_argument("-m", "--models", required=True,
                help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
                help="# of models to train")
args = vars(ap.parse_args())

# Load the training and testing data, then scale it into the range [0, 1]
print("[INFO]: Loading CIFAR-10 data....")
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# loop over the number of models to train
for i in np.arange(0, args["num_models"]):
	# initialize the optimizer and model
	print("[INFO] training model {}/{}".format(i+1, args["num_models"]))
	optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
	model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

	# train the network
	H = model.fit_generator(aug.flow(train_x, train_y, batch_size=32), 
		validation_data=(test_x, test_y), steps_per_epoch=len(train_x)//32, 
		epochs=40, verbose=1)

	# save the model to disk
	p = [args["models"], "model_{}.model".format(i)]
	model.save(os.path.sep.join(p))

	# evaluate the network
	# Evaluate the network
	predictions = model.predict(test_x, batch_size=32)
	report = classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)

	# save the classification report to file
	p = [args["output"], "model_{}.txt".format(i)]
	f = open(os.path.sep.join(p), "w")
	f.write(report)
	f.close()

	# plot the training loss and accuracy
	p = [args["output"], "model_{}.png".format(i)]
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy for model {}".format(i))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(os.path.sep.join(p))
	plt.close()
