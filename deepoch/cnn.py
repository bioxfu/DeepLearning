from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling2D

class FCHeadNet:
	@staticmethod
	def build(modelName, baseModel, classes, D):
		# initialize the head model that will be placed on top of
		# the base, then add a FC layer
		headModel = baseModel.output

		if modelName in ['VGG16', 'VGG19']:
			headModel = Flatten(name="flatten")(headModel)
		elif modelName in ['InceptionV3', 'Xception', 'ResNet50']:
			headModel = GlobalAveragePooling2D()(headModel)

		headModel = Dense(D, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)

		# add a softmax layer
		headModel = Dense(classes, activation="softmax")(headModel)

		# return the model
		return headModel

