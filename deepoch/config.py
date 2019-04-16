# define the paths to the images directory
IMAGES_PATH = '../datasets/PlantVillage/train'

# fine-tuning model
#PRE_TRAIN_MODELS = ['VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'Xception']
#PRE_TRAIN_MODELS = ['VGG16']
#PRE_TRAIN_MODELS = ['InceptionV3']
PRE_TRAIN_MODELS = ['ResNet50']

# resize image keeping aspect ratio
IMAGES_SIZE = {'VGG16': 224, 'VGG19': 224, 'ResNet50': 224, 'InceptionV3': 299, 'Xception': 299}

# take a number of images from the training data and 
# use them as validation and test data
NUM_CLASSES = 38
NUM_VAL_IMAGES = 5000
NUM_TEST_IMAGES = 5000

# define the path to the output training, validation, 
# and testing HDF5 files
TRAIN_HDF5 = '../datasets/PlantVillage/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/PlantVillage/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/PlantVillage/hdf5/test.hdf5'

# define the path to the dataset mean which is used to store
# the average red, green and blue pixel intensity value across
# the entire training dataset
DATASET_MEAN = 'output/dataset_RGB_mean.json'

# define the path to the output directory
OUTPUT_PATH = 'output'

BATCH_SIZE = 4

