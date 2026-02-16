import torch
from datasets import load_from_disk 
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
import collections
import Config 

#===========================================================================
# Functions directly for the Tokenizer and text processing
#===========================================================================

# ---------------------------------------------------------
# 1. SHARED PREPROCESSING FUNCTION
# ---------------------------------------------------------
def preprocess_text(text):
    """
    Function takes the text in the string format and performs these operations:
    
    1. Undercapitalize
    
    2. Collapse repeated punctuation (!!! -> !), but keeping multiple dots (...)
    
    3. Separate punctuation into distinct tokens. This allows for better token saving.
    Instead of saving words like (cat!, cat, cat? , cat;  dog!, dog, dog? ...)
    We save them as cat, dog and the punctiactions !,?,; which are shared across whole dataset
    
    4. Split by whitespace
    """
    
    if not text:
        return []

    #Undercapitalize
    text = text.lower()
    
    #Capturing repeating punctiations and collapsing them into single one
    text = re.sub(r'([!?,])\1+', r'\1', text)
    
    #Splitting the punctuations into separate "words" - thatnks to that they are treated as separate token
    text = re.sub(r'([.,!?;:()"\-+=_])', r' \1 ', text)
    
    return text.split()


class SimpleTokenizer:
    """
    Tokenizer which, well tokenizes the words into the numbers.
    Now supports OFFLINE LEMMATIZATION via a static map lookup.
    """
    
    def __init__(self, vocab_path, lemma_map_path=None, max_length=None):
        
        #Opening created vocab.json file
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Load Lemma Map if provided (Offline Lemmatization)
        self.lemma_map = {}
        if lemma_map_path and os.path.exists(lemma_map_path):
            with open(lemma_map_path, 'r', encoding='utf-8') as f:
                self.lemma_map = json.load(f)
        
        #Parameter of max sequence length - calculated empirically for the problem
        self.max_length = max_length
        
        #Invert vocabulary for the optimziation in the lookup - in the decode
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        
        #Unknown and padding vocab ids
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def encode(self, text):
        # 1. Preprocess the words (Clean & Split)
        words = preprocess_text(text)
        
        # 2. Offline Lemmatization
        # If the word exists in our lemma_map, replace it with the base form.
        if self.lemma_map:
            words = [self.lemma_map.get(w, w) for w in words]

        # 3. Map to IDs
        tokens = [self.vocab.get(w, self.unk_id) for w in words]
        
        # 4. Processing of the token list (padding/cropping)
        if self.max_length:
            tokens = tokens[:self.max_length]
            tokens += [self.pad_id] * (self.max_length - len(tokens))
            
        return tokens

    def decode(self, token_ids, skip_special=True):
        #Processing the tokens into the words
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()
        
        if isinstance(token_ids[0], list):
            return [self._decode_single(seq, skip_special) for seq in token_ids]
        
        return self._decode_single(token_ids, skip_special)

    def _decode_single(self, ids, skip_special):
        #Single sequence processing. Mapping tokens into the words
        
        words = []
        for idx in ids:
            idx = int(idx)
            #Getting words. If not in the dict then give UNKNOWN word
            word = self.inverse_vocab.get(idx, "<UNK>")
            
            #Skpiing special tokens (only the pad, as its unnecssary to show it)
            if skip_special:
                if idx == self.pad_id:
                    continue
            
            words.append(word)
        
        return " ".join(words)


#===========================================================================
# Functions to create the vocabulary for the tokenizer
#===========================================================================


def _worker_scan(args):
    """
    Scans dataset to collect RAW Word frequencies (before lemmatization).
    We do NOT lemmatize here to keep workers fast and dependency-free.
    """
    
    dataset_path, start_idx, end_idx, batch_size = args
    dataset = load_from_disk(dataset_path)
    
    local_word_counts = collections.Counter()
    local_len_counts = collections.Counter() # Tracks distribution of lengths
    
    for i in range(start_idx, end_idx, batch_size):
        slice_end = min(i + batch_size, end_idx)
        batch = dataset[i : slice_end]
        
        batch_tokens = []
        batch_lengths = []

        # Helper to process a single string
        def process_entry(txt):
            if txt:
                tokens = preprocess_text(txt)
                batch_tokens.extend(tokens)
                batch_lengths.append(len(tokens))

        # 1. Primary Captions
        for text in batch['caption']:
            process_entry(text)
        
        # 2. Augmented Captions
        if 'caption_aug' in batch:
            for aug_list in batch['caption_aug']:
                if aug_list:
                    for text in aug_list:
                        process_entry(text)
                        
        local_word_counts.update(batch_tokens)
        local_len_counts.update(batch_lengths)
        
    return local_word_counts, local_len_counts


def plot_rank_frequency_comparison(raw_counts, lemma_counts, min_frequency, save_path):
    """
    Overlays Raw vs. Lemmatized frequency distributions to show the 'Lift' effect.
    """
    sorted_raw = sorted(raw_counts.values(), reverse=True)
    sorted_lemma = sorted(lemma_counts.values(), reverse=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot Raw (Base)
    plt.plot(sorted_raw, color='gray', linestyle=':', linewidth=1.5, label='Raw Words (Pre-Lemma)')
    
    # Plot Lemmatized (Improved)
    plt.plot(sorted_lemma, color='darkblue', linewidth=2.0, label='Lemmatized (Post-Lemma)')
    
    # Cutoff Line
    plt.axhline(y=min_frequency, color='red', linestyle='--', label=f'Cutoff ({min_frequency})')
    
    plt.title(f'Impact of Lemmatization on Vocabulary Size\nRaw Unique: {len(sorted_raw)} | Lemma Unique: {len(sorted_lemma)}')
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_caption_lengths(length_counts, save_path):
    """
    Function just for visualisation of the word length distribution
    For visual inspection
    """
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    
    #Percentiles for the 95 and 99% cutoffs
    total_samples = sum(counts)
    cumulative = np.cumsum(counts)
    p95 = lengths[np.searchsorted(cumulative, 0.95 * total_samples)] if lengths else 0
    p99 = lengths[np.searchsorted(cumulative, 0.99 * total_samples)] if lengths else 0
    max_len = lengths[-1] if lengths else 0

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, color='forestgreen', alpha=0.7, width=1.0)
    
    #Percentiles lines
    plt.axvline(x=p95, color='orange', linestyle='--', label=f'95% Coverage ({p95})')
    plt.axvline(x=p99, color='red', linestyle='--', label=f'99% Coverage ({p99})')
    
    plt.title(f'Caption Length Distribution (Max: {max_len})')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Captions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    
    plt.savefig(save_path)
    plt.close()
    print(f"Stats: 95% of captions are <= {p95} tokens, 99% <= {p99} tokens")


def create_vocabulary(arrow_path, min_frequency=5, save_vocab_path="vocab.json", save_map_path="lemma_map.json", num_workers=None):
    """
    Creates both the Lemma Map (offline lookup) and the Vocabulary in a single run.
    """
    
    print(f"Loading dataset from {arrow_path}...")
    dataset = load_from_disk(arrow_path)
    total_len = len(dataset)
    
    #If num workers is not specified it takes it with leaving range for the system to work
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    print(f"Phase 1: Scanning Raw Words with {num_workers} workers...")
    
    #Splitting work to the chunks for the worekrs
    chunk_size = total_len // num_workers
    worker_args = []
    
    #Start and end different for each worker, giving them work to do. They are workers
    for i in range(num_workers):
        start = i * chunk_size
        end = total_len if i == num_workers - 1 else (i + 1) * chunk_size
        worker_args.append((arrow_path, start, end, 5000))

    global_raw_words = collections.Counter()
    global_lengths = collections.Counter()
    
    
    #Start work, then merge the results 
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_worker_scan, worker_args), total=num_workers, desc="Scanning Raw"))
        
        print("Merging raw results...")
        for w_count, l_count in results:
            global_raw_words.update(w_count)
            global_lengths.update(l_count)


    # -------------------------------------------------------------
    # Phase 2: Build Lemma Map using SpaCy (Main Process Only)
    # -------------------------------------------------------------
    print("\nPhase 2: Building Lemma Map (Spacy)...")
    try:
        import spacy
        # Load lightweight model
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except ImportError:
        print("Error: SpaCy not installed. Cannot build lemma map.")
        return
    except OSError:
        print("Error: 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        return

    unique_raw_words = list(global_raw_words.keys())
    lemma_map = {}
    
    # Batch process for speed
    batch_size = 5000
    for i in tqdm(range(0, len(unique_raw_words), batch_size), desc="Lemmatizing Unique Words"):
        batch = unique_raw_words[i : i + batch_size]
        docs = list(nlp.pipe(batch))
        
        for original, doc in zip(batch, docs):
            lemma = doc[0].lemma_.lower().strip()
            # Only save if different (compression)
            if lemma != original and lemma:
                lemma_map[original] = lemma

    # Save Lemma Map
    with open(save_map_path, 'w', encoding='utf-8') as f:
        json.dump(lemma_map, f, ensure_ascii=False, indent=4)
    print(f"Lemma map saved to {save_map_path} (Size: {len(lemma_map)})")


    # -------------------------------------------------------------
    # Phase 3: Aggregate Counts for Vocab
    # -------------------------------------------------------------
    print("\nPhase 3: Building Final Vocab from Lemmas...")
    global_lemma_counts = collections.Counter()
    
    for raw_word, count in global_raw_words.items():
        # Map raw -> lemma. If not in map, use raw.
        final_word = lemma_map.get(raw_word, raw_word)
        global_lemma_counts[final_word] += count

    #Create the plots (Using the comparison version)
    plot_rank_frequency_comparison(global_raw_words, global_lemma_counts, min_frequency, "Plots/vocab_rank_freq.png")
    plot_caption_lengths(global_lengths, "Plots/caption_length_hist.png")


    #Based on the analysis (it was in fact firtly runned to just see params and then adjusted by plots)
    #Filtering the words
    print(f"Filtering words < {min_frequency}...")
    filtered_vocab = {w: c for w, c in global_lemma_counts.items() if c >= min_frequency}
    sorted_words = sorted(filtered_vocab.items(), key=lambda item: item[1], reverse=True)
    
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    
    #Saving the vocab.json
    with open(save_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
        
    print(f"Done. Final Vocab Size: {len(vocab)}")
    return vocab



#===========================================================================
#Running the script and creating vocabulary
#===========================================================================
if __name__ == "__main__":
    path = os.path.join(Config.DATABASE_PATH)
    
    # Ensure plots directory exists
    os.makedirs("Plots", exist_ok=True)
    
    # (1)
    #Creating Vocabulary and the Lemma Map
    create_vocabulary(path, 
                      min_frequency=2, 
                      save_vocab_path="vocab.json",
                      save_map_path="lemma_map.json",
                      num_workers=5 
                      )

    
    #Simple tests of the tokenizer
    print("\nTesting Tokenizer\n")
    
    # Now that files are created, we can initialize the tokenizer
    tokenizer = SimpleTokenizer(vocab_path="vocab.json", 
                                lemma_map_path="lemma_map.json", 
                                max_length=128) 
    
    #Test
    sample_text = 'Two dogs are running fast!' 
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded) 
    
    print(f"Input:   {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")