import argparse
from keras.preprocessing.image import ImageDataGenerator

### Parse command-line arguments
def options(x='test'):
    ap = argparse.ArgumentParser()
    if x == 'common':
        ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
        ap.add_argument("-o", "--output", required=True, help="path to the output directory")
        ap.add_argument("-w", "--weights", required=True, help="path to the best model weights file")
        ap.add_argument("-e", "--epochs", required=True, type=int, help="number of epoches")
        ap.add_argument("-s", "--image-size", required=True, type=int, help="size of images")
        ap.add_argument("-b", "--batch-size", required=True, type=int, help="batch size of images to be passed through network")
    elif x == 'test':
        ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
        ap.add_argument("-s", "--image-size", required=True, type=int, help="size of images")
        ap.add_argument("-b", "--batch-size", required=True, type=int, help="batch size of images to be passed through network")
        ap.add_argument("-w", "--weights", required=True, help="path to the best model weights file")
    elif x == 'inspect_model_layers':
        ap.add_argument("-m", "--model", type=str, required=True, help="name of pre-trained network to use (ResNet50, InceptionV3, Xception, VGG16, VGG19)")
        ap.add_argument("-p", "--plot", type=str, required=True, help="plot of model")
        ap.add_argument("-i", "--include-top", type=int, default=1, help="whether (1) or not (-1) to include top of CNN")


    args = vars(ap.parse_args())
    return args

def image_augment():
    aug = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1, 
                             shear_range=0.2, 
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")
    return aug

