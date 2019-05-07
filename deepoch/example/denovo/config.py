# define the paths to the images directory
IMAGES_PATH = '../datasets/root_shoot/root/image'
LABEL_DIR = -2

# fine-tuning model
TRAIN_MODELS = ['RootNet']

# resize image keeping aspect ratio
DB_IMAGES_SIZE = 42
IMAGES_SIZE = {'RootNet': 32}

# take a number of images from the training data and 
# use them as validation and test data
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1000
NUM_TEST_IMAGES = 1000

# define the path to the output training, validation, 
# and testing HDF5 files
TRAIN_HDF5 = '../datasets/root_shoot/root/hdf5/train.hdf5'
VAL_HDF5 = '../datasets/root_shoot/root/hdf5/val.hdf5'
TEST_HDF5 = '../datasets/root_shoot/root/hdf5/test.hdf5'

# define the path to the dataset mean which is used to store
# the average red, green and blue pixel intensity value across
# the entire training dataset
DATASET_MEAN = 'output/dataset_RGB_mean.json'

# define the path to the output directory
OUTPUT_PATH = 'output'

CLASS_NAMES = 'output/class_names'

# hyperparameters
BATCH_SIZE = 64
#LEARNING_RATE = 1e-3
LEARNING_RATE = 1e-6
EPCHO = 100
