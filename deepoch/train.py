from tools import options, image_augment
from dataIO import HDF5DatasetGenerator
from cnn import FCHeadNet
from imutils import paths
from preprocessor import SimplePreprocessor
from preprocessor import PatchPreprocessor
from preprocessor import MeanPreprocessor
from preprocessor import ImageToArrayPreprocessor
import config
import json
from finetune import finetune
import os

def tain():
    pre_train_models = config.PRE_TRAIN_MODELS
    image_size = config.IMAGES_SIZE
    batch_size = config.BATCH_SIZE
    output_path = config.OUTPUT_PATH
    # construct the training image generator for data augmentation
    aug = image_augment()
    
    # load the RGB means for the training set
    means = json.loads(open(config.DATASET_MEAN).read())

    for pre_train_model in pre_train_models:
    
        saved_model = os.path.sep.join([output_path, '{}_model.hdf5'.format(pre_train_model)])

        # initialize the image preprocessors
        sp = SimplePreprocessor(image_size[pre_train_model], image_size[pre_train_model])
        pp = PatchPreprocessor(image_size[pre_train_model], image_size[pre_train_model])
        mp = MeanPreprocessor(means['R'], means['G'], means['B'])
        itap = ImageToArrayPreprocessor()

        # initialize the training and validation dataset generators
        trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batchSize=batch_size, aug=aug,
            preprocessors=[pp, mp, itap], classes=config.NUM_CLASSES)
        valGen = HDF5DatasetGenerator(config.VAL_HDF5, batchSize=batch_size,
            preprocessors=[sp, mp, itap], classes=config.NUM_CLASSES)

        finetune(pre_train_model, trainGen, valGen, batch_size, output_path, saved_model)

        trainGen.close()
        valGen.close()

    if debug == True:
        pass

    
if __name__ == '__main__':

    tain()
