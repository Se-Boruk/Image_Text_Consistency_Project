import torch
import time
import random
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import math
import re
import random
import torch
import time


# 1. GPU Session Options
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ONNX_MODEL_PATH = "./models/Qwen2.5-1.5B_onnx"

print("🚀 Loading Qwen2.5-1.5B onto GPU (CUDA)...")
tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH)

# 2. Initialize with CUDA Provider
# use_io_binding=True is critical for GPU performance
model = ORTModelForCausalLM.from_pretrained(
    ONNX_MODEL_PATH,
    file_name="model_fp16.onnx",
    provider="CUDAExecutionProvider",
    use_io_binding=True, 
    session_options=sess_options,
    provider_options={
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 12 * 1024 * 1024 * 1024, # Adjust based on your VRAM
    }
)




def generate_positive_shortened(caption):
    # 1. Count Sentences
    # We split by common sentence enders (., !, ?)
    sentences = [s.strip() for s in re.split(r'[.!?]+', caption) if s.strip()]
    num_sentences = len(sentences)
    
    # 2. Determine target count based on your distribution
    # 50% chance: Half (rounded up)
    # 30% chance: ~75% length
    # 20% chance: ~25% (or at least 1)
    rand = random.random()
    if rand < 0.45:
        target_count = max(1, math.ceil(num_sentences * 0.50))
        mode = "HALF"
    elif rand < 0.80:
        target_count = max(1, math.ceil(num_sentences * 0.7))
        mode = "MOST"
    else:
        target_count = max(1, math.ceil(num_sentences * 0.3))
        mode = "MINIMAL"

    # Calculate token limits roughly based on sentence average
    avg_words_per_sent = len(caption.split()) / max(1, num_sentences)
    target_tokens = int(avg_words_per_sent * target_count)

    # 3. Structural Prompt
    # We tell the model specifically how many sentences to keep.
    
    #Randomization of prompt
    # 3. Double Randomizer: Temperature & Top_P
    # Low temp = literal; High temp = varied phrasing
    dyn_temp = random.uniform(0.05, 0.15) 
    dyn_top_p = random.uniform(0.75, 0.8)
    dyn_rep_p = random.uniform(0.9, 1.15)

    # 4. Prompt with varied starting styles to prevent "1)" or "In this image"
    prompt = f"Task: Shorten the original caption. Do not add new adjectives, motivations, narratives or things non-existing in the original.\nOriginal: {caption}\nShortened ({target_count} sentence):"


    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids_len = inputs.input_ids.shape[1]
    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=target_tokens + 15,
            do_sample=True,
            temperature=dyn_temp,
            top_k = 5,
            top_p=dyn_top_p,
            repetition_penalty=dyn_rep_p, # Higher penalty stops it from copying "In this image" from prompt
            # Stop tokens
            eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n")[0]],
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True)
    # Post-process to remove unwanted numbering or meta-talk
    clean = response.splitlines()[0].strip()
    clean = re.sub(r'^\d+[\).]\s*', '', clean) # Removes "1)" or "1."

    return clean, time.perf_counter() - start_time, f"T={dyn_temp:.2f} | {mode}"


# --- GPU SPEED TEST ---
test_captions = [
    "In this picture we can see there are two wash basins on the bathroom vanity. Behind the bathroom vanity, there is a mirror.",
    "Here in this picture we can see a red colored table present and beside that we can see yellow colored doors present all over there.",
    "In the image we can see this is a luggage bag, Inside the luggage bag there is a cat.",
    "This image is taken in a tennis court. In this image there are two women playing tennis with tennis bat. In the background there is a net. In the bottom of the image there is a dustbin and three empty chairs.",
    "A man is standing on the left wearing a specs. A woman is standing holding something in the hand. On the floor there is a footwear and a mat. Also there are some items in the background. In the back there is a table and a chair. Behind him there is a sofa. On the right side there are windows with curtain. And a serial lights around the window.",
    "In this image I can see a woman standing. She is holding a tennis ball and a tennis racket. This is the tennis net. In the background, I can see trees and hills."

]


attempts = 10

for p in range(attempts):
    print("-" * 80)
    print("Attempt:", p)
    for cap in test_captions:
        rewritten, duration, task_str = generate_positive_shortened(cap)
        print(f"Original: {cap}")
        print(f"\nRewritten: {rewritten}")
        print(f"⏱️ GPU Latency: {duration:.4f}s")
        print("-" * 40)
    
    
    
    
    
    
    
    
    