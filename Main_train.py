###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

from DataBase_functions import Custom_DataSet_Manager
from DataBase_functions import Async_DataLoader
import torch
from tqdm import tqdm

import DataBase_functions as d_func

import Config

###################################################################
# ( 1 ) Hardware setup
###################################################################

print("\nSearching for cuda device...")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print available GPUs
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


###################################################################
# ( 2 ) Loading data
###################################################################

TRAIN_SPLIT = Config.TRAIN_SPLIT
VAL_SPLIT = Config.VAL_SPLIT
TEST_SPLIT = Config.TEST_SPLIT

RANDOM_STATE = Config.RANDOM_STATE
DATABASE_PATH = Config.DATABASE_PATH

#Load manager and execute
manager = Custom_DataSet_Manager(DataSet_path = DATABASE_PATH,
                                 train_split = TRAIN_SPLIT,
                                 val_split = VAL_SPLIT,
                                 test_split = TEST_SPLIT,
                                 random_state = RANDOM_STATE
                                 )


#Load dataset
Train_set, Val_set, Test_set = manager.load_dataset_from_disk()

#Verify stratified splits (different datasets could cause not optimal learning. Theyre stratified by the source)
print("="*60)
print("\nStratification test:")
d_func.verify_splits(Train_set, Val_set, Test_set)


print("="*60)
print("\nHash verification if dataset splits are the same across runs:")
d_func.verify_dataset_integrity(Train_set, Val_set, Test_set, Config.SPLIT_HASHES)

###################################################################
# ( 3 ) Setting parameters
###################################################################

EPOCHS = Config.EPOCHS
BATCH_SIZE = Config.BATCH_SIZE
TOKEN_LENGTH = Config.TOKEN_LENGTH

N_WORKERS = Config.N_WORKERS
MAX_QUEUE = Config.MAX_QUEUE
TRAIN_SET_FRACTION = Config.TRAIN_SET_FRACTION


###################################################################
# ( 4 ) Model creation, dataloader preparation
###################################################################

# Training loader
train_loader = Async_DataLoader(dataset = Train_set,
                                batch_size=BATCH_SIZE,
                                token_length = TOKEN_LENGTH,
                                num_workers=N_WORKERS,
                                device='cuda',
                                max_queue=MAX_QUEUE,
                                add_augmented = True,
                                fraction = TRAIN_SET_FRACTION
                                )

# Validation loader
val_loader = Async_DataLoader(dataset = Val_set,
                                batch_size=BATCH_SIZE,
                                token_length = TOKEN_LENGTH,
                                num_workers=N_WORKERS,
                                device='cuda',
                                max_queue=MAX_QUEUE,
                                add_augmented = True,
                                fraction = TRAIN_SET_FRACTION
                                )

# Test loader
test_loader = Async_DataLoader(dataset = Test_set,
                                batch_size=BATCH_SIZE,
                                token_length = TOKEN_LENGTH,
                                num_workers=N_WORKERS,
                                device='cuda',
                                max_queue=MAX_QUEUE,
                                add_augmented = True,
                                fraction = TRAIN_SET_FRACTION
                                )


jghjgh

for e in range(EPOCHS):
    #################################################
    #Training part
    #################################################
    
    train_loader.start_epoch(shuffle=True)
    
    epoch_loss = 0.0
    num_batches = train_loader.get_num_batches()

    with tqdm(total=num_batches, desc=f"Epoch {e+1}", unit=" batch") as pbar:
        while True:
            #Load batch from loader
            batch = train_loader.get_batch()
            if batch is None:
                break
            
            
            ############
            #TRAIN FUNCTION
            #########
            # update progressbar/time
            Loss = 0.5
            pbar.update(1)
            pbar.set_postfix( { "train_loss": f"{Loss:.4f}" } )




"""
import matplotlib.pyplot as plt

origin = batch["origin"]
image_original = batch['image_original']
image_augmented = batch['image_augmented']
caption_positive = batch['caption_positive']
caption_negeative = batch['caption_negeative']



image_augmented = image_augmented.permute(0,2,3,1).cpu().numpy()
image_original = image_original.permute(0,2,3,1).cpu().numpy()


i = 17
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image_original[i])

plt.subplot(1,2,2)
plt.title("Augmented")
plt.imshow(image_augmented[i])
"""



