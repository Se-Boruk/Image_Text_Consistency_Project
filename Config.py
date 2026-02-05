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

SOURCE_MAP = {
              "mscoco_train2017": 0,
              "ade20k": 1,
              "flick30k": 2
              }

SPLIT_HASHES = {
    "train": "7686595d368b2c9647cc9c77e168d700c0ead13e808de647ac09af2864108f53",
    "val": "fa6f29ac8d62881053f9d57f5c65a94405f96644a28d5b95736f244d9ce2a2f7",
    "test": "dfad87ad4a698f84bcdfaa1bd61c833e9bfbfe7d9f6d5d693e30db1c115a3d9d"
}


#Training hyperparams
########################################################

EPOCHS = 2
BATCH_SIZE = 32
TOKEN_LENGTH = 128


N_WORKERS = 5
MAX_QUEUE = 10
TRAIN_SET_FRACTION = 1



