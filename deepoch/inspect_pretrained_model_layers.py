from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception  # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.utils import plot_model
import argparse
from tools import options
import sys

args = options('inspect_model_layers')

# Define a dictionary that maps model names to their classes inside Keras
# See all the availabel models: https://keras.io/applications/
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,  # TensorFlow ONLY
    "resnet": ResNet50
}

print("[INFO] loading network....")
if args['model'] == 'VGG16':
	model = VGG16(weights="imagenet", include_top=args["include_top"] > 0)
elif args['model'] == 'VGG19':
	model = VGG19(weights="imagenet", include_top=args["include_top"] > 0)
elif args['model'] == 'ResNet50':
	model = ResNet50(weights="imagenet", include_top=args["include_top"] > 0, input_shape=(224, 224, 3))
elif args['model'] == 'InceptionV3':
	model = InceptionV3(weights="imagenet", include_top=args["include_top"] > 0)
elif args['model'] == 'Xception':
	model = Xception(weights="imagenet", include_top=args["include_top"] > 0)
else:
	sys.exit()

print("[INFO] showing layers of {}....".format(args['model']))
# loop over the layers in the network and display them
for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

plot_model(model, to_file=args['plot'], show_shapes=True)
