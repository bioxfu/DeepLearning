from keras.layers.core import Dropout, Flatten, Dense, Activation
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K

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


class RootNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # Initialize the model, input shape and the channel dimension
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        # If we are using 'channels_first', update the input shape and channels dimension
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dim = 1

        # Block #1: first CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), strides=(1, 1),
            padding='valid', input_shape=input_shape,
            kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),
            padding='valid', input_shape=input_shape,
            kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(128, (3, 3), strides=(1, 1), 
        	padding='valid', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), 
        	padding='valid', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: second CONV => RELU => POOL layer set
        model.add(Conv2D(256, (3, 3), strides=(1, 1), 
        	padding='valid', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), 
        	padding='valid', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))

        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #5: second set of FC => RELU layers
        model.add(Dense(128, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))

        # Return the constructed network architecture
        return model
