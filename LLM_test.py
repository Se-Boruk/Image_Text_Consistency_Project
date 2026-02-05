import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import random


# --- CONFIG ---
# Zmień tę ścieżkę na folder, w którym masz zapisany model
LOCAL_MODEL_PATH = "./models/Qwen2.5-1.5B" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sprawdzenie wsparcia dla BF16
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# Hard Test: Sprawdzenie czy folder istnieje
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"❌ Nie znaleziono modelu w: {LOCAL_MODEL_PATH}. Pobierz go najpierw lub popraw ścieżkę.")

print(f"🚀 Device: {DEVICE} | Dtype: {DTYPE}")

# --- LOAD MODEL (LOCAL) ---
# Podając ścieżkę do folderu zamiast ID z HuggingFace, Transformers czytają pliki lokalnie
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)

# Ręczne dodanie brakującego szablonu SmolLM2 (format Jinja)
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )




model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH, 
    torch_dtype=DTYPE, 
    device_map=DEVICE,
    local_files_only=True
)

def generate_negative_caption(original_caption):
    mode = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
    
    # Precyzyjne instrukcje systemowe dla 1.5B
    if mode == 0:
        task = "Change one color or number. Keep the rest of the sentence identical."
    elif mode == 1:
        task = "Add one weird accessory (like sunglasses, a crown, or holding a stick) to the subject."
    else:
        task = "Change the location or the main action to something slightly different."

    # --- CHAT TEMPLATE (Klucz do logiki 1.5B) ---
    messages = [
        {"role": "system", "content": "You are a precise image description editor. Output ONLY the modified sentence."},
        {"role": "user", "content": f"Task: {task}\nOriginal: {original_caption}"}
    ]
    
    # To doda tagi <|im_start|> i <|im_end|>, co "budzi" logikę modelu
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Fix dla ostrzeżenia o pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=60, # 1.5B potrzebuje trochę więcej "oddechu" na logikę
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    latency = time.time() - start_time
    # Dekodujemy TYLKO to, co model wygenerował (pomijając prompt)
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Czyszczenie
    response = response.split("\n")[0].strip()
    if "." in response:
        response = response.split(".")[0] + "."
        
    return response, latency, mode

# --- TESTY ---
if __name__ == "__main__":
    caps = [
        "A man in a red hat walking a golden retriever.",
        "Three birds sitting on a wooden fence during a bright sunset."
    ]
    for c in caps:
        for _ in range(2):
            neg, lat, m = generate_negative_caption(c)
            print(f"\n[MODE {m}] {c} \n-> {neg}")