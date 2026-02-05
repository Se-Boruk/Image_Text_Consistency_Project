from datasets import load_dataset, load_from_disk
import os
import numpy as np
import torch
from queue import Queue
import threading
import random
import math
import torch.nn.functional as F
import torch
from PIL import Image as PILImage
from collections import Counter
from datasets import ClassLabel
import hashlib


from Config import SOURCE_MAP




class DummyTokenizer:
    def __init__(self, max_length=128):
        self.max_length = max_length

    def __call__(self, caption):
        """
        Converts a list of strings into a LongTensor of ASCII values.
        """
        tokens = torch.zeros((self.max_length), dtype=torch.long)
        
        return tokens




def verify_splits(train, val, test):
    for name, ds in [("Train", train), ("Val", val), ("Test", test)]:
        counts = Counter(ds['dataset_source'])
        total = len(ds)
        print(f"\n{name} Distribution ({total} samples):")
        for src, count in counts.items():
            print(f" - {src}: {count} ({count/total:.2%})")

def verify_dataset_integrity(train_ds, val_ds, test_ds, expected_hashes=None):

    
    samples = {
        "train": str(train_ds[0]['caption']),
        "val": str(val_ds[0]['caption']),
        "test": str(test_ds[0]['caption'])
    }
    
    current_hashes = {}
    
    for split_name, text in samples.items():
        #hASH FOR CAPTION
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        current_hashes[split_name] = h

        
    if expected_hashes:
        mismatch = False
        for split in ["train", "val", "test"]:
            if current_hashes[split] != expected_hashes.get(split):
                print(f"ERROR: {split} hash mismatch!")
                mismatch = True
        
        if not mismatch:
            print("All hashes match. Dataset splits are consistent.")
        else:
            raise ValueError("Integrity check failed! Splits are different than expected.")
    else:
        print("No expected hashes provided. Copy the values above to Config")
        for split, h in current_hashes.items():
            print(f"{split.capitalize()} first sample hash: {h}")
        



def preprocess_image_to_square(img_input, target_size=256):
    """
    Universal preprocessor for PIL, NumPy, or Torch inputs.
    Resizes maintaining aspect ratio and pads to a square.
    """
    
    #COnversion logic
    if isinstance(img_input, torch.Tensor):
        # Handle (C, H, W) or (H, W, C) tensors
        
        if img_input.ndimension() == 3 and img_input.shape[0] in [1, 3]:
            img_input = img_input.permute(1, 2, 0)
        img_input = img_input.cpu().detach().numpy()

    if isinstance(img_input, np.ndarray):
        #Ensure uint8 for PIL conversion
        if img_input.max() <= 1.0:
            img_input = (img_input * 255).astype(np.uint8)
        img = PILImage.fromarray(img_input)
    else:
        # Assume already a PIL image or try to force it
        img = img_input

    #Standardization to RGB
    img = img.convert("RGB")
    
    #Geometry Logic
    width, height = img.size
    scale = target_size / max(width, height)
    new_width, new_height = int(width * scale), int(height * scale)
    
    img = img.resize((new_width, new_height), PILImage.LANCZOS)
    
    #Canvas Creation
    final_img = PILImage.new("RGB", (target_size, target_size), (0, 0, 0))
    upper = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    final_img.paste(img, (left, upper))
    
    return final_img



class Custom_DataSet_Manager():
    
    #Checks if there is dataset folder present, if not it creates it
    def __init__(self, DataSet_path, train_split, val_split, test_split, random_state):
        self.dataset_path = DataSet_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        

    def load_dataset_from_disk(self):  
        #Load it to split it on run
        Dataset = load_from_disk(self.dataset_path)
        
        train, val, test = self.split_dataset(Dataset)
        return train, val, test
    
    def split_dataset(self,dataset):

        
        #takint unique classes and print them
        unique_sources = dataset.unique("dataset_source")
        print(f"Detected unique sources: {unique_sources}")
        
        #Encoding data source to number so it can be used in stratification
        dataset = dataset.map(
                lambda x: {"dataset_source": [s.lower() for s in x["dataset_source"]]},
                batched=True,
                desc="dataset_source casing to "
            )
        
        sorted_names = [name.lower() for name, _ in sorted(SOURCE_MAP.items(), key=lambda x: x[1])]
        
        source_feature = ClassLabel(names=sorted_names)
        dataset = dataset.cast_column("dataset_source", source_feature)
        
        print(f"Mapowanie klas : {dataset.features['dataset_source']._str2int}")
        
        ####
        Data =  dataset.shuffle(seed=self.random_state)
        
        #Split it into train and subset
        split_dataset = Data.train_test_split(test_size= (1 -self.train_split) , seed=self.random_state, stratify_by_column="dataset_source")
        
        train_subset = split_dataset['train']
        subset = split_dataset['test']
        
        #Split the subset into the val and test 
        test_fraction = self.val_split / ((self.val_split + self.test_split))
        
        split_dataset_1 = subset.train_test_split(test_size= test_fraction , seed=self.random_state, stratify_by_column="dataset_source")
        
        val_subset = split_dataset_1['train']
        test_subset = split_dataset_1['test']
        
        
        return train_subset, val_subset, test_subset
        

    
##########################################################################    
    
    
class Async_DataLoader():
    def __init__(self, dataset, batch_size=32,token_length = 128, num_workers=2, device='cuda', max_queue=10, add_augmented = True, fraction = None):
        self.dataset = dataset
        #Taking sample of from dataset to initialize the shape of images
        sample_img = np.array(dataset[0]["image"], dtype=np.uint8)
        self.C, self.H, self.W = sample_img.shape[2], sample_img.shape[0], sample_img.shape[1]
        
        self.fraction = fraction # Fraction of images taken in the epoch (randomized each epoch)
        
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.queue = Queue(maxsize=max_queue)
        self.num_workers = num_workers

        #Epoch control
        self.next_idx = 0               #Next step (batch) idx
        self.idx_lock = threading.Lock()
        self.active_workers = 0 
        self.threads = []
        self.epoch_event = threading.Event()
        self.indices = list(range(len(self.dataset)))
        

        self.token_length = token_length
        #Preallocate pinned buffers
        self.pinned_bufs = [torch.empty((self.batch_size, self.C, self.H, self.W), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        self.caption_bufs = [torch.empty((self.batch_size,self.token_length), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        
        self.origin_bufs = [torch.empty((self.batch_size,1), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        self.add_augmented = add_augmented
        
        self.tokenizers = [DummyTokenizer(self.token_length) for _ in range(self.num_workers)]
        
        
        # threads will be started lazily in start_epoch (safer on Windows/spawn)
        self.threads_started = False
        
        # do not start prefetch here
        # self._start_prefetch()

    def _start_prefetch(self):
        
        def get_chunk():
            with self.idx_lock:
                start = self.next_idx
                if start >= self.effective_len:  # use effective length
                    return None, None
                
                end = min(start + self.batch_size, self.effective_len)  # use effective length
                self.next_idx = end
                return start, end


        def worker(worker_id):
            pinned_buf = self.pinned_bufs[worker_id]
            caption_bufs = self.caption_bufs[worker_id]
            origin_bufs = self.origin_bufs[worker_id]
            
            Tokenizer = self.tokenizers[worker_id]
        
            while True:
                self.epoch_event.wait()  # wait for epoch start
        
                while True:
                    start, end = get_chunk()
                    if start is None:
                        break
                    actual_bs = end - start
        
                    # ---------------------------
                    # Load original batch into pinned memory
                    # ---------------------------
                    for i in range(actual_bs):
                        idx = self.indices[start + i]
                        
                        img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
                        
                        caption = self.dataset[idx]['caption']
                        caption = Tokenizer(caption)

                        origin = self.dataset[idx]['dataset_source']
                        
                        if origin == "coco":
                            origin = 0
                        elif origin == "ade20k":
                            origin = 1
                        elif origin == "flick30k":
                            origin = 2
                        else:
                            origin = 3
                        
                        ######
                        pinned_buf[i].copy_(torch.from_numpy(img).permute(2, 0, 1))
                        caption_bufs[i].copy_(caption)
                        origin_bufs[i].fill_(origin)
        
                    # Clone to avoid modifying pinned memory
                    origin_batch = origin_bufs[:actual_bs].to(self.device, non_blocking=True).clone()
                    original_batch = pinned_buf[:actual_bs].to(self.device, non_blocking=True).clone()
                    original_captions = caption_bufs[:actual_bs].to(self.device, non_blocking=True).clone()
                    
        
                    # ---------------------------
                    # Prepare batch variants
                    # ---------------------------
                    batch_dict = {
                        'origin': origin_batch,
                        "image_original": original_batch,
                        "caption_positive": original_captions
                    }
        
                    # Augmented
                    if self.add_augmented:
                        aug_images = augment_images(original_batch)
                        batch_dict["image_augmented"] = aug_images
                    
                    #Prepare the negative caption as well - currently just the placeholder
                    ###################################
                    batch_dict["caption_negeative"] = original_captions
                    ###################################
                    
                    # Push to queue
                    self.queue.put(batch_dict)
        
                # Epoch end handling
                with self.idx_lock:
                    self.active_workers -= 1
                    if self.active_workers == 0:
                        self.queue.put(None)
                        self.epoch_event.clear()

        # start worker threads
        for wid in range(self.num_workers):
            t = threading.Thread(target=worker, args=(wid,))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def start_epoch(self, shuffle=True):
        self.queue = Queue(maxsize=self.queue.maxsize)
        self.next_idx = 0
        self.active_workers = self.num_workers
    
        if not self.threads_started:
            self._start_prefetch()
            self.threads_started = True
    
        # Shuffle and optionally reduce dataset fraction
        indices = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(indices)
    
        if self.fraction is not None and 0 < self.fraction < 1:
            reduced_size = int(len(indices) * self.fraction)
            
            indices = np.random.choice(indices, size=reduced_size, replace=False)
    
        self.indices = list(indices)
        self.effective_len = len(self.indices)  #store effective lenght
    
        self.epoch_event.set()
        
        


    def get_batch(self):
        batch = self.queue.get()
        if batch is None:
            return None
    
        # Move all tensors in dict to device (non-blocking)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def get_num_batches(self):
        if hasattr(self, "effective_len"):
            effective_len = self.effective_len
        else:
            effective_len = len(self.dataset)
        steps = (effective_len + self.batch_size - 1) // self.batch_size
        return steps


    def get_random_batch(self, batch_size=None, shuffle=True, random_state=None):
        """
        Returns a random batch of original images from the dataset, reproducible if rng is provided.
        
        Args:
            batch_size: int, optional
            shuffle: bool, whether to pick random indices
            rng: np.random.Generator, optional — if given, sampling becomes deterministic
    
        Returns:
            batch tensor of shape (B, C, H, W)
        """
        bs = batch_size or self.batch_size
    
        #Random state for reproducibility
        if random_state is None:
            random_state = np.random.default_rng()
        else:
            random_state = np.random.default_rng(random_state)

        
        # pick indices
        if shuffle:
            indices = random_state.choice(len(self.dataset), bs, replace=False)
        else:
            indices = np.arange(bs)
    
        #preallocate pinned buffer
        pinned_buf = torch.empty((bs, self.C, self.H, self.W),
                                 dtype=torch.float32).pin_memory()
    
        #load images
        for i, idx in enumerate(indices):
            img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
            pinned_buf[i] = torch.from_numpy(img).permute(2, 0, 1)
    
        #move to device
        batch = pinned_buf.to(self.device, non_blocking=True)
        
        return batch
    
    
##########################################################################


def augment_images(image_batch,
                    brightness=0.2, contrast=0.2, saturation=0.2,
                    flip_prob=0.5, max_rot=15, crop_ratio=0.9):
    
    """
    image_batch: (B, C, H, W) tensor, float in [0,1]
    Returns: augmented_image_batch, (in [0,1])
    """
    B, C, H, W = image_batch.shape
    device = image_batch.device
    dtype = image_batch.dtype


    # clone to avoid in-place modifications
    image_batch = image_batch.clone()

    # -----------------------------
    # 1. Random horizontal flip
    # -----------------------------
    flip_mask = torch.rand(B, device=device) < flip_prob
    if flip_mask.any():
        image_batch[flip_mask] = image_batch[flip_mask].flip(dims=[-1])


    # -----------------------------
    # 2. Random rotation
    # -----------------------------
    angles = (torch.rand(B, device=device) * 2 - 1) * max_rot  # -max_rot .. +max_rot
    radians = angles * (3.14159265 / 180)

    cos = torch.cos(radians)
    sin = torch.sin(radians)
    rot_matrices = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    rot_matrices[:, 0, 0] = cos
    rot_matrices[:, 0, 1] = -sin
    rot_matrices[:, 1, 0] = sin
    rot_matrices[:, 1, 1] = cos
    rot_matrices[:, :, 2] = 0  # rotate around center

    grid = F.affine_grid(rot_matrices, image_batch.size(), align_corners=False)
    image_batch = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # -----------------------------
    # 3. Random crop + resize
    # -----------------------------
    if crop_ratio < 1.0:
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        top = torch.randint(0, H - crop_h + 1, (B,), device=device)
        left = torch.randint(0, W - crop_w + 1, (B,), device=device)

        cropped_images = torch.zeros_like(image_batch)

        for i in range(B):
            cropped_images[i] = F.interpolate(
                image_batch[i:i+1, :, top[i]:top[i]+crop_h, left[i]:left[i]+crop_w],
                size=(H, W), mode='bilinear', align_corners=False
            )

        image_batch = cropped_images

    # -----------------------------
    # 4. Color jitter
    # -----------------------------
    b_factors = 1.0 + (torch.rand(B,1,1,1, device=device, dtype=dtype) * 2 - 1) * brightness
    image_batch = image_batch * b_factors

    mean = image_batch.mean(dim=[2,3], keepdim=True)
    c_factors = 1.0 + (torch.rand(B,1,1,1, device=device, dtype=dtype) * 2 - 1) * contrast
    image_batch = (image_batch - mean) * c_factors + mean

    if C == 3:
        gray = image_batch.mean(dim=1, keepdim=True)
        s_factors = 1.0 + (torch.rand(B,1,1,1, device=device, dtype=dtype) * 2 - 1) * saturation
        image_batch = (image_batch - gray) * s_factors + gray

    # -----------------------------
    # 5. Clamp to [0,1] range
    # -----------------------------
    image_batch = image_batch.clamp(0, 1)


    return image_batch

    