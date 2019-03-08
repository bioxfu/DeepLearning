from os import path

# define the paths to the training and validation directory
TRAIN_IMAGES = "/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/train"
VAL_IMAGES = "/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/val/images"

# define the paths to the file that maps validation filenames to
# their corresponding class labels
VAL_MAPPINGS = "/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/val/val_annotations.txt"

# define the paths to the WordNet hierarchy files which are used
# to generate our class labels
WORDNET_IDS = "/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/words.txt"

# since we do not have access to the testing data we need to 
# take a number of images from the training data and use them instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = '/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/hdf5/train.hdf5'
VAL_HDF5 = '/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/hdf5/val.hdf5'
TEST_HDF5 = '/home/xfu/Git/DeepLearning/myTools/datasets/tiny-imagenet-200/hdf5/test.hdf5'

# define the path to the dataset mean
DATASET_MEAN = '/home/xfu/Git/DeepLearning/myTools/project/deepergooglenet/output/tiny-imagenet-200-mean.json'

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = '/home/xfu/Git/DeepLearning/myTools/project/deepergooglenet/output'
MODEL_PATH = path.sep.join([OUTPUT_PATH, "checkpoints/weights.008-4.1598.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "deepergooglenet_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "deepergooglenet_tinyimagenet.json"])

