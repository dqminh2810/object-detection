# Holds our configuration settings, which will be used in our selection of Python scripts
# TODO

# import the necessary packages
import os

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories
ORIG_BASE_PATH = "raw-data"
# fork class
FORK_BASE_PATH =  os.path.sep.join([ORIG_BASE_PATH, "fork"])
FORK_ORIG_IMAGES = os.path.sep.join([FORK_BASE_PATH, "images"])
FORK_ORIG_ANNOTS = os.path.sep.join([FORK_BASE_PATH, "pascal"])
# knife class
KNIFE_BASE_PATH =  os.path.sep.join([ORIG_BASE_PATH, "knife"])
KNIFE_ORIG_IMAGES = os.path.sep.join([KNIFE_BASE_PATH, "images"])
KNIFE_ORIG_ANNOTS = os.path.sep.join([KNIFE_BASE_PATH, "pascal"])
# plate class
PLATE_BASE_PATH =  os.path.sep.join([ORIG_BASE_PATH, "plate"])
PLATE_ORIG_IMAGES = os.path.sep.join([PLATE_BASE_PATH, "images"])
PLATE_ORIG_ANNOTS = os.path.sep.join([PLATE_BASE_PATH, "pascal"])



# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "dataset"
TRAINING_BASE_PATH = os.path.sep.join([BASE_PATH, "training"])
SIMPLE_TRAINING_BASE_PATH = os.path.sep.join([TRAINING_BASE_PATH, "simple"])
SS_TRAINING_BASE_PATH = os.path.sep.join([TRAINING_BASE_PATH, "search_selective"])

# fork class
FORK_TRAINING_BASE_PATH = os.path.sep.join([SS_TRAINING_BASE_PATH, "fork"])
SIMPLE_FORK_TRAINING_BASE_PATH = os.path.sep.join([SIMPLE_TRAINING_BASE_PATH, "fork"])
SIMPLE_NO_FORK_TRAINING_BASE_PATH = os.path.sep.join([SIMPLE_TRAINING_BASE_PATH, "no_fork"])

# knife class
KNIFE_TRAINING_BASE_PATH = os.path.sep.join([SS_TRAINING_BASE_PATH, "knife"])
SIMPLE_KNIFE_TRAINING_BASE_PATH = os.path.sep.join([SIMPLE_TRAINING_BASE_PATH, "knife"])
SIMPLE_NO_KNIFE_TRAINING_BASE_PATH = os.path.sep.join([SIMPLE_TRAINING_BASE_PATH, "no_knife"])

# plate class
PLATE_TRAINING_BASE_PATH = os.path.sep.join([SS_TRAINING_BASE_PATH, "plate"])
SIMPLE_PLATE_TRAINING_BASE_PATH = os.path.sep.join([SIMPLE_TRAINING_BASE_PATH, "plate"])
SIMPLE_NO_PLATE_TRAINING_BASE_PATH = os.path.sep.join([SIMPLE_TRAINING_BASE_PATH, "no_plate"])



# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 5
MAX_NEGATIVE = 5

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)
# define the path to the output model and label binarizer
MODEL_PATH = "object_detector_simple.h5"
ENCODER_PATH = "label_encoder_simple.pickle"
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.98