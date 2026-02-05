import torch
import time
import random
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# --- Performance Optimization ---
sess_options = ort.SessionOptions()
# Optimal for modern workstations: set to number of physical cores
sess_options.intra_op_num_threads = 4  
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Path to the FP16 ONNX export
ONNX_MODEL_PATH = "./models/Qwen2.5-1.5B_onnx"

print("🚀 Loading Qwen2.5-1.5B...")
tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH)

# Load FP16 Model (Ensure model.onnx and model.onnx_data are in the folder)
model = ORTModelForCausalLM.from_pretrained(
    ONNX_MODEL_PATH,
    file_name="model.onnx", # Standard name for FP16/FP32 export
    provider="CPUExecutionProvider",
    session_options=sess_options
)

def get_randomized_task():
    """Builds a probabilistic task code with specific logic blocks."""
    rule_configs = [
        (lambda n: f"ADD_{n}_OBJECTS", 0.30, 1, 2),
        (lambda n: f"CHANGE_{n}_ATTRIBUTES", 0.40, 1, 3),
        (lambda n: f"SWAP_SUBJECT", 0.20, 1, 1),
        (lambda n: f"IMPOSSIBLE_ACTION", 0.10, 1, 1),
        (lambda n: f"REMOVE_{n}_DETAILS", 0.15, 1, 2)
    ]
    
    active = []
    for func, prob, min_n, max_n in rule_configs:
        if random.random() < prob:
            active.append(func(random.randint(min_n, max_n)))
    
    if not active: return get_randomized_task()
    return " + ".join(active)

def generate_dynamic_negative(caption):
    task_code = get_randomized_task()
    words = caption.split()
    
    # Anchor length 3-4 works best for long context to prevent meta-talk
    anchor_len = 3 if len(words) > 3 else 1
    anchor = " ".join(words[:anchor_len])

    # --- THE HARD-RAIL PROMPT ---
    # Schema-driven few-shot is mandatory to keep 1.5B models from drifting.
    prompt = (
        "TASK: Transform the INPUT into a factually INCORRECT version.\n"
        "EXAMPLE 1:\nInput: A cat on a mat.\nTask: SWAP_SUBJECT\nOutput: A dog on a mat.\n\n"
        "EXAMPLE 2:\nInput: Two blue cars parked.\nTask: CHANGE_2_ATTRIBUTES\nOutput: Three red cars parked.\n\n"
        f"INPUT: {caption}\n"
        f"RULES: {task_code}\n"
        "MANDATE: Output ONLY the sentence. No 'In this task'. No 'Inaccurate Version'.\n"
        f"OUTPUT: {anchor}"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_len = inputs.input_ids.shape[1]

    # Dynamic scaling: scaling output relative to input length
    l_random = random.uniform(0.7, 1.2)
    max_new = int(len(words) * l_random) + 8

    start_time = time.time()
    
    # Deterministic Greedy Search for stability
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        repetition_penalty=1.8, # Prevents copying input or repeating 'Task'
        no_repeat_ngram_size=3,  # Hard stop on repetitive phrases
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    latency = time.time() - start_time
    response = tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True)
    
    # Cleaning: Remove any meta-hallucinations or trailing thoughts
    clean = response.split('\n')[0].split('.')[0].strip()
    
    # Defensive programming: filter common meta-prefixes
    meta_prefixes = ["Output:", "Negative:", "Result:", "The incorrect version is:"]
    for prefix in meta_prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()

    final_output = f"{anchor} {clean}."
    return final_output, latency, task_code

# --- DIVERSE TEST ---
test_captions = [
    "A man in a red hat walking a golden retriever.",
    "A shiny blue sports car parked on a busy city street.",
    "A young woman reading a book in a quiet library.",
    "There is a young man walking with his dog. The sun is shining and there is a street nearby. One of the truck is red.",
    "In this picture we can see there are two wash basins on the bathroom vanity. Behind the bathroom vanity, there is a mirror.",
    "In this picture I can see the inside view of a room, there is a bed with pillows and blankets on it, there are lamps."
]

print("-" * 80)
for cap in test_captions:
    negative, duration, task_str = generate_dynamic_negative(cap)
    print(f"Original: {cap}")
    print(f"Strategy: {task_str}")
    print(f"Negative: {negative}")
    print(f"⏱️ Time: {duration:.4f}s")
    print("-" * 40)