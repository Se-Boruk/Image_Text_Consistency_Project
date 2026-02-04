import os


#Basic params and variabels
########################################################
DATABASE_RAW_PATH = os.path.join("Database", "raw")
DATABASE_PATH = os.path.join("Database", "processed")

RANDOM_STATE = 111

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

DEFAULT_IMG_SIZE = 224



#Training hyperparams
########################################################

EPOCHS = 2
BATCH_SIZE = 32
TOKEN_LENGTH = 128


N_WORKERS = 5
MAX_QUEUE = 10
TRAIN_SET_FRACTION = 1



