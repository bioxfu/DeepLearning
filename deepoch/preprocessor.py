import cv2
import imutils
import numpy as np
from keras.preprocessing.image import img_to_array
from sklearn.feature_extraction.image import extract_patches_2d


class SimplePreprocessor:
    '''Resize the image to a fixed size ignoring the aspect ratio.'''
    
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        '''
        :param width: The target width of input image after resizing
        :param height: The target height of input image after resizing
        :param interpolation: Interpolation algorithm used when resizing
        '''
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        '''
        :param image: Image
        :return: Re-sized image
        '''
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)


class AspectAwarePreprocessor:
    '''Resize the image to a fixed size maintaining the aspect ratio.'''
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimentions of the image and then initialize
        # the deltas to use when cropping
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # if the width is smaller than the height, the resize
        # along the width (i.e., the smaller dimension) and then
        # update the deltas to crop the height to the desired
        # demension
        if w < h:
            image = imutils.resize(image, width=self.width,
                inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # to crop along the width
        else:
            image = imutils.resize(image, height=self.width,
                inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # re-grab the width and height, followd by preforming 
        # the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # due to rounding errors, we need to ensure out output
        # image has the desired width and height
        return cv2.resize(image, (self.width, self.height), 
            interpolation=self.inter)


class ImageToArrayPreprocessor:
    '''Accept an input image and then properly orders the channels based on 
    image_data_format setting'''
    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


class MeanPreprocessor:
    '''Subtract the mean Red, Green, and Blue pixel intensities 
    across a dataset from an input image which is a form of data
    normalization'''
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue
        # channels
        # OpenCV represents images in BGR order rather than RGB
        (B, G, R) = cv2.split(image.astype("float32"))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channels back together and return the image
        return cv2.merge([B, G, R])


class PatchPreprocessor:
    '''Crop a random portion of the input image and pass it to the network.
    It's a very effective method to avoid overfitting by applying another 
    layer of data augmentation''' 
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract a random crop from the image with the target width
        # and height
        return extract_patches_2d(image, (self.height, self.width),
            max_patches=1)[0]


class CropPreprocessor:
    '''During the evaluating phase of CNN, we crop the four corners of 
    the input image + the center region and then take their corresponding
    horizontal flips, for a total of ten samples per input image.
    These ten samples will be passed through the CNN, and then the 
    probabilites averaged.'''
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image then use these
        # dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
                    [0, 0, self.width, self.height],
                    [w - self.width, 0, w, self.height],
                    [w - self.width, h - self.height, w, h],
                    [0, h - self.height, self.width, h]
                 ]

        # compute teh center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # loop over the coordinates, extract each of the crops, 
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height),
                interpolation=self.inter)
            crops.append(crop)

        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        # return the set of crops
        return np.array(crops)

