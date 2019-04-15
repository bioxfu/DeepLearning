import argparse
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

def rank_accuracy(preds, labels):
    # initialzie the rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0

    # loop over the predictions and ground-truth labels
    for (p, gt) in zip(preds, labels):
        # sort the probabilites by their index in descending
        # order to that the more confident guesses are at the
        # front of the list
        p = np.argsort(p)[::-1]

        # check if the ground-truth label is in the top-5
        # predictions
        if gt in p[:5]:
            rank5 += 1

        # check if the ground-truth label is the #1 prediction
        if gt == p[0]:
            rank1 += 1

    # compute the finak rank-1 and rank-5 accuracies
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))
    
    # return a tuple of the rank-1 and rank-5 accuracies
    return(rank1, rank5)
