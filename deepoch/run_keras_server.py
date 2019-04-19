# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import config
from preprocessor import ImageToArrayPreprocessor
from preprocessor import AspectAwarePreprocessor
from preprocessor import MeanPreprocessor
from preprocessor import CropPreprocessor
from dataIO import HDF5DatasetGenerator
from keras.models import load_model
from PIL import Image
import numpy as np
import progressbar
import json
import flask
import os
import io

# http://codeleading.com/article/9038704258/
import tensorflow as tf
graph = tf.get_default_graph()


pre_train_model = config.PRE_TRAIN_MODELS[0]
image_size = config.IMAGES_SIZE[pre_train_model]
output_path = config.OUTPUT_PATH
saved_model = os.path.sep.join([output_path, '{}_model.hdf5'.format(pre_train_model)])

f = open(config.CLASS_NAMES)
class_names = [x.strip() for x in f.readlines()]
f.close()

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
aap = AspectAwarePreprocessor(image_size, image_size)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
cp = CropPreprocessor(image_size, image_size)
itap = ImageToArrayPreprocessor()

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_saved_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = load_model(saved_model)

def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	
	image = np.asarray(image)
	image = aap.preprocess(image)
	image = mp.preprocess(image)
	crops = cp.preprocess(image)
	crops = np.array([itap.preprocess(c) for c in crops], dtype="float32")

	# return the processed image
	return crops

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		print(flask.request)
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image)

			# classify the input image and then initialize the list
			# of predictions to return to the client
			with graph.as_default():
				y_prob = model.predict(image)
			y_prob = y_prob.mean(axis=0)

			# sorting the predictions in descending order
			sorting = (-y_prob).argsort()
			# getting the top 5 predictions
			sorted_ = sorting[:5]

			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for value in sorted_:
				label = class_names[value]
				prob = y_prob[value]
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_saved_model()
	app.run(host='0.0.0.0')

