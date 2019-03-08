from keras.applications import VGG16
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1,
                help="whether (1) or not (-1) to include top of CNN")
args = vars(ap.parse_args())

# load the VGG16 network
print("[INFO] loading network....")
model = VGG16(weights="imagenet",
	include_top=args["include_top"] > 0)

print("[INFO] showing layers....")
# loop over the layers in the network and display them
# to the console
for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
