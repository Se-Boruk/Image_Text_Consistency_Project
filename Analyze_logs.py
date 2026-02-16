import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "Plots/training_log.csv" 
OUTPUT_DIR = "Plots"

def analyze_training_logs():

    if not os.path.exists(CSV_PATH):

        if os.path.exists("training_log.csv"):
            file_path = "training_log.csv"
        else:
            print(f"Error: Could not find log file at '{CSV_PATH}' or 'training_log.csv'.")
            return
    else:
        file_path = CSV_PATH

    print(f"Reading log file: {file_path}")

    df = pd.read_csv(file_path)

    #Make dir if not existing
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    #==========================================
    # Mertrics
    #==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training vs Validation Metrics', fontsize=16)

    #Loss
    ax = axes[0, 0]
    ax.plot(df['Epoch'], df['T_Loss'], label='Train Loss', color='darkblue', linewidth=2)
    ax.plot(df['Epoch'], df['V_Loss'], label='Val Loss', color='indianred', linewidth=2, linestyle='--')
    ax.set_title('Loss')
    ax.set_ylabel('Loss Value')
    ax.legend()
    ax.grid(True)

    #Balanced accuracy
    ax = axes[0, 1]
    ax.plot(df['Epoch'], df['T_B_Acc'], label='Train B-Acc', color='darkblue', linewidth=2)
    ax.plot(df['Epoch'], df['V_B_Acc'], label='Val B-Acc', color='indianred', linewidth=2, linestyle='--')
    ax.set_title('Balanced Accuracy')
    ax.set_ylabel('Balanced Accuracy')
    ax.legend()
    ax.grid(True)

    #Recall
    ax = axes[1, 0]
    ax.plot(df['Epoch'], df['T_Recall'], label='Train Recall', color='darkblue', linewidth=2)
    ax.plot(df['Epoch'], df['V_Recall'], label='Val Recall', color='indianred', linewidth=2, linestyle='--')
    ax.set_title('Recall')
    ax.set_ylabel('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    #Specificity
    ax = axes[1, 1]
    ax.plot(df['Epoch'], df['T_Spec'], label='Train Spec', color='darkblue', linewidth=2)
    ax.plot(df['Epoch'], df['V_Spec'], label='Val Spec', color='indianred', linewidth=2, linestyle='--')
    ax.set_title('Specificity')
    ax.set_ylabel('Specificity')
    ax.legend()
    ax.grid(True, alpha=0.3)


    plt.tight_layout()
    save_path_metrics = os.path.join(OUTPUT_DIR, 'metrics_over_training.png')
    plt.savefig(save_path_metrics, dpi=150)
    plt.close()


    #==========================================
    #Treshold
    #==========================================
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['Epoch'], df['Thresh'], label='Optimal Threshold', color='darkblue', linewidth=2, marker='o', markersize=4)
    

    plt.title('Decision Threshold Calibration over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Threshold Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path_thresh = os.path.join(OUTPUT_DIR, 'threshold_over_training.png')
    plt.savefig(save_path_thresh, dpi=150)
    plt.close()


if __name__ == "__main__":
    analyze_training_logs()




