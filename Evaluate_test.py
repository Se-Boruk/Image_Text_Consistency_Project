import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import Config
import Architectures as Arch
from DataBase_functions import Custom_DataSet_Manager, Async_DataLoader


#Settings params and hardware
MODEL_NAME = "best_model.pth" 
CHECKPOINT_PATH = os.path.join("Models", "Trained", MODEL_NAME)
PLOT_SAVE_DIR = "Plots"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("\nLoading Test Data...")
manager = Custom_DataSet_Manager(
    DataSet_path=Config.DATABASE_PATH,
    train_split=Config.TRAIN_SPLIT,
    val_split=Config.VAL_SPLIT,
    test_split=Config.TEST_SPLIT,
    random_state=Config.RANDOM_STATE
)


_, _, Test_set = manager.load_dataset_from_disk()

test_loader = Async_DataLoader(
    dataset=Test_set,
    batch_size=Config.BATCH_SIZE,
    sequence_length=Config.SEQUENCE_LENGTH,
    num_workers=Config.N_WORKERS,
    device='cuda',
    max_queue=Config.MAX_QUEUE,
    image_augmentation=False,
    fraction=1
)


##########################################
#Model setup and loading
##########################################
print(f"\nLoading Model: {MODEL_NAME}...")

text_model = Arch.Text_encoder(
    vocab_size=Config.VOCAB_SIZE,
    word_dim=Config.TOKEN_DIM,
    hidden_dim=Config.HIDDEN_DIM_LSTM,
    embed_dim=Config.LATENT_SPACE,
    depth=Config.LSTM_DEPTH
)

image_model = Arch.Image_encoder(
    embed_dim=Config.LATENT_SPACE,
    weights_path = "None"
)

model = Arch.Siamese_model(
    Image_model=image_model,
    Text_model=text_model,
    device=device
)


##########################################
#Loading weights - wit fallbacks
##########################################
if os.path.exists(CHECKPOINT_PATH):
    try:
        # Try loading normally
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Standard load failed ({e}), attempting with safe globals...")
        try:
            import numpy as np
            try:
                from numpy._core.multiarray import scalar
            except ImportError:
                from numpy.core.multiarray import scalar
            
            torch.serialization.add_safe_globals([scalar])
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        except Exception as e2:
            raise RuntimeError(f"Could not load checkpoint. Error: {e2}")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Weights loaded successfully.")
else:
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

model.move_to_device()
model.eval_mode()

##########################################
#Inference loop
##########################################
pos_scores = []
neg_scores = []

print("\nRunning Inference on Test Set...")
num_batches = test_loader.get_num_batches()
test_loader.start_epoch(shuffle=False)

with torch.no_grad():
    with tqdm(total=num_batches, desc="Evaluating", unit="batch") as pbar:
        while True:
            batch = test_loader.get_batch()
            if batch is None: break

            img = batch['image_original']
            pos_cap = batch['caption_positive']
            neg_cap = batch['caption_negative']

            _, sim_pos = model.predict(img, pos_cap, threshold=0.5)
            _, sim_neg = model.predict(img, neg_cap, threshold=0.5)

            pos_scores.append(sim_pos.cpu())
            neg_scores.append(sim_neg.cpu())
            
            pbar.update(1)

pos_scores = torch.cat(pos_scores).numpy()
neg_scores = torch.cat(neg_scores).numpy()

##########################################
#Treshold analysis
##########################################
print("\nCalculating Metrics across thresholds...")

thresholds = np.linspace(0.0, 1.0, 101)
b_accs = []
recalls = []
specificities = []
diffs = [] 

for t in thresholds:
    tp = np.sum(pos_scores > t)
    fn = np.sum(pos_scores <= t)
    tn = np.sum(neg_scores <= t)
    fp = np.sum(neg_scores > t)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    b_acc = (recall + specificity) / 2
    
    diff = abs(recall - specificity)

    recalls.append(recall)
    specificities.append(specificity)
    b_accs.append(b_acc)
    diffs.append(diff)


##########################################
#Finding best thresholds for given conditions (specificity, recall and bal_acc)
##########################################

#Max Balanced Accuracy
max_acc = max(b_accs)
idx_max = b_accs.index(max_acc)
t_max_acc = thresholds[idx_max]

#Equilibrium (same recall and spec)
min_diff = min(diffs)
idx_eq = diffs.index(min_diff)
t_equilibrium = thresholds[idx_eq]

print("\n" + "="*40)
print(f"RESULTS FOR {MODEL_NAME}")
print("="*40)
print(f"1. Peak Balanced Acc Threshold: {t_max_acc:.2f}")
print(f"   -> B-Acc: {b_accs[idx_max]:.4f}")
print(f"   -> Recall: {recalls[idx_max]:.4f}")
print(f"   -> Spec:   {specificities[idx_max]:.4f}")
print("-" * 40)
print(f"2. Equilibrium Threshold:       {t_equilibrium:.2f}")
print(f"   -> B-Acc: {b_accs[idx_eq]:.4f}")
print(f"   -> Recall: {recalls[idx_eq]:.4f}")
print(f"   -> Spec:   {specificities[idx_eq]:.4f}")
print("="*40)

##########################################
#Plotting
##########################################
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
plot_filename = f"Threshold_Analysis_{MODEL_NAME.replace('.pth', '')}.png"
save_path = os.path.join(PLOT_SAVE_DIR, plot_filename)

plt.style.use('bmh')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Threshold Analysis: {MODEL_NAME}', fontsize=16)


# --- Helper Function for Dynamic Legends ---
def draw_detailed_lines(ax, val_at_max, val_at_eq, metric_name):
    """
    Draws vertical lines and includes the specific metric value in the legend
    """
    
    label_max = f'Peak B-Acc T:{t_max_acc:.2f} ({metric_name}={val_at_max:.3f})'
    ax.axvline(t_max_acc, color='black', linestyle='--', alpha=0.6, label=label_max)

    label_eq = f'Equilibrium T:{t_equilibrium:.2f} ({metric_name}={val_at_eq:.3f})'
    ax.axvline(t_equilibrium, color='blue', linestyle='-.', alpha=0.6, label=label_eq)
    
    ax.legend(loc='lower left', fontsize=9)


#Bal acc
ax = axes[0, 0]
ax.plot(thresholds, b_accs, color='#2ca02c', linewidth=2)
ax.set_title('Balanced Accuracy')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')

draw_detailed_lines(ax, b_accs[idx_max], b_accs[idx_eq], "B-Acc")
ax.grid(True, alpha=0.3)

#Recall vs spec
ax = axes[0, 1]
ax.plot(thresholds, recalls, label='Recall', color='#9467bd', linewidth=2)
ax.plot(thresholds, specificities, label='Specificity', color='#e377c2', linewidth=2)
ax.set_title('Recall vs Specificity (Crossing Point)')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')

#Legend
rec_at_max = recalls[idx_max]
spec_at_max = specificities[idx_max]
ax.axvline(t_max_acc, color='black', linestyle='--', alpha=0.6, 
           label=f'Peak B-Acc T:{t_max_acc:.2f}\n(Rec={rec_at_max:.2f}, Spec={spec_at_max:.2f})')

rec_at_eq = recalls[idx_eq]
spec_at_eq = specificities[idx_eq]
ax.axvline(t_equilibrium, color='blue', linestyle='-.', alpha=0.6, 
           label=f'Equilibrium T:{t_equilibrium:.2f}\n(Rec≈Spec≈{rec_at_eq:.2f})')
ax.legend(loc='lower left', fontsize=8)
ax.grid(True, alpha=0.3)

#Recall 
ax = axes[1, 0]
ax.plot(thresholds, recalls, color='#9467bd', linewidth=2)
ax.set_title('Recall (Sensitivity)')
ax.set_xlabel('Threshold')

draw_detailed_lines(ax, recalls[idx_max], recalls[idx_eq], "Rec")
ax.grid(True, alpha=0.3)

#Specificity
ax = axes[1, 1]
ax.plot(thresholds, specificities, color='#e377c2', linewidth=2)
ax.set_title('Specificity (True Negative Rate)')
ax.set_xlabel('Threshold')

draw_detailed_lines(ax, specificities[idx_max], specificities[idx_eq], "Spec")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to: {save_path}")
plt.show()



