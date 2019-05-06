# define the paths to the images directory
IMAGES_PATH = '../datasets/BreakHis/BreaKHis_v1/histology_slides/breast'
LABEL_DIR = -2

# fine-tuning model
#TRAIN_MODELS = ['VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'Xception']
TRAIN_MODELS = ['VGG16']
#TRAIN_MODELS = ['InceptionV3']
#TRAIN_MODELS = ['ResNet50']

# resize image keeping aspect ratio
DB_IMAGES_SIZE = 300
IMAGES_SIZE = {'VGG16': 224, 'VGG19': 224, 'ResNet50': 224, 'InceptionV3': 299, 'Xception': 299}

# take a number of images from the training data and 
# use them as validation and test data
NUM_CLASSES = 2
NUM_VAL_IMAGES = 200
NUM_TEST_IMAGES = 200

# define the path to the output training, validation, 
# and testing HDF5 files
TRAIN_HDF5 = '../datasets/BreakHis/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/BreakHis/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/BreakHis/hdf5/test.hdf5'

# define the path to the dataset mean which is used to store
# the average red, green and blue pixel intensity value across
# the entire training dataset
DATASET_MEAN = 'output/dataset_RGB_mean.json'

# define the path to the output directory
OUTPUT_PATH = 'output'

CLASS_NAMES = 'output/class_names'

# hyperparameters
BATCH_SIZE = 4
LEARNING_RATE_SHALLOW = 1e-4
LEARNING_RATE_DEEP = 1e-6
EPCHO_SHALLOW = 10
EPCHO_DEEP = 100
