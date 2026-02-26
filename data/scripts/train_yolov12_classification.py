from ultralytics import YOLO
import torch
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import re


# ---------------------------------------------------------
# CONFIG
def get_model_version(project_dir, model_base):
    if not os.path.exists(project_dir):
        return f"{model_base}_v1"

    existing = os.listdir(project_dir)
    versions = []

    pattern = re.compile(rf"{model_base}_v(\d+)")

    for name in existing:
        match = pattern.match(name)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1
    return f"{model_base}_v{next_version}"

# ---------------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------------
CLASS_NAMES = [
    "bird_drop",
    "clean",
    "dusty",
    "electrical_damage",
    "physical_damage",
    "snow_covered"
]

# ---------------------------------------------------------
# 🔥 YENİ: CLASS IMBALANCE ANALYSİS
# ---------------------------------------------------------
def analyze_class_distribution(dataset_path, split="train"):
    """Sınıf dağılımını analiz eder"""
    print(f"\n[INFO] Analyzing class distribution in {split} set...")
    class_counts = {}
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_path, split, class_name)
        count = len(glob.glob(os.path.join(class_dir, "*")))
        class_counts[class_name] = count
    
    total = sum(class_counts.values())
    print(f"\nTotal {split} images: {total}")
    for cls, cnt in class_counts.items():
        print(f"  {cls:20s}: {cnt:4d} ({cnt/total*100:.1f}%)")
    
    return class_counts


def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    DATASET_PATH = "data/classification/rgb"
    MODEL_NAME = "data/models/yolo8m-cls.pt"
    PROJECT_DIR = "runs/classification"
    MODEL_BASE = os.path.splitext(os.path.basename(MODEL_NAME))[0]
    EXP_NAME = get_model_version(PROJECT_DIR, MODEL_BASE)

    EPOCHS = 100
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LR = 1e-3
    PATIENCE = 15
    DEVICE = 0 if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Tüm setleri analiz et
    # ---------------------------------------------------------
    train_dist = analyze_class_distribution(DATASET_PATH, "train")
    val_dist = analyze_class_distribution(DATASET_PATH, "val")
    test_dist = analyze_class_distribution(DATASET_PATH, "test")

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    print("\n[INFO] Loading YOLO classification model...")
    model = YOLO(MODEL_NAME)

    # ---------------------------------------------------------
    # 🔥 YENİ: ADVANCED TRAINING WITH AUGMENTATION
    # ---------------------------------------------------------
    print("\n[INFO] Starting training with advanced settings...")
    train_results = model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        
        # Optimizer settings
        lr0=LR,
        lrf=0.01,  # 🔥 Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        optimizer="AdamW",
        
        # 🔥 Data Augmentation (Güneş panelleri için optimize edilmiş)
        hsv_h=0.015,        # Hue augmentation
        hsv_s=0.5,          # Saturation
        hsv_v=0.4,          # Value/Brightness
        degrees=15.0,       # Rotation ±15 derece
        translate=0.1,      # Translation
        scale=0.3,          # Scale variation
        shear=0.0,          # Shear (paneller için gereksiz)
        perspective=0.0,    # Perspective (güneş panelleri için az)
        flipud=0.0,         # Vertical flip (paneller için mantıksız)
        fliplr=0.5,         # Horizontal flip (%50 şans)
        mosaic=0.0,         # Classification için mosaic kapalı
        mixup=0.1,          # 🔥 MixUp augmentation
        copy_paste=0.0,     # Classification için gereksiz
        
        # Regularization
        dropout=0.2,        # 🔥 Dropout for overfitting prevention
        
        # Training settings
        patience=PATIENCE,
        device=DEVICE,
        val=True,
        save_period=10,     # Her 10 epoch'ta checkpoint kaydet
        
        # Outputs
        project=PROJECT_DIR,
        name=EXP_NAME,
        save=True,
        verbose=True,
        plots=True,
        
        # 🔥 Learning rate scheduler
        cos_lr=True,        # Cosine annealing scheduler
        
        # 🔥 Close mosaic (son epochlarda augmentation azalt)
        close_mosaic=10,
    )

    print("\n[INFO] Training completed.")

    # ---------------------------------------------------------
    # 🔥 YENİ: BEST MODEL LOADING
    # ---------------------------------------------------------
    print("\n[INFO] Loading best model from training...")
    best_model_path = f"{PROJECT_DIR}/{EXP_NAME}/weights/best.pt"
    model = YOLO(best_model_path)

    # ---------------------------------------------------------
    # TEST / EVALUATION
    # ---------------------------------------------------------
    print("\n[INFO] Running evaluation on TEST set...")
    metrics = model.val(
        data=DATASET_PATH,
        split="test",
        plots=True,
        save_json=True,
    )

    print(f"\n{'='*50}")
    print(f"Top-1 Accuracy : {metrics.top1:.4f}")
    print(f"{'='*50}\n")

    # ---------------------------------------------------------
    # CONFUSION MATRIX + CLASSIFICATION REPORT
    # ---------------------------------------------------------
    print("[INFO] Computing confusion matrix and classification report...")

    y_true = []
    y_pred = []

    test_root = os.path.join(DATASET_PATH, "test")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        image_paths = glob.glob(os.path.join(test_root, class_name, "*"))
        print(f"Processing {class_name}: {len(image_paths)} images...")
        
        for img_path in image_paths:
            # Normal prediction
            pred = model.predict(img_path, verbose=False, augment=False)[0]
            y_true.append(class_idx)
            y_pred.append(pred.probs.top1)
            
    # ---------------------------------------------------------
    # 🔥 YENİ: COMPARE NORMAL
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Test Score")
    print("="*50)

    from sklearn.metrics import accuracy_score

    acc_normal = accuracy_score(y_true, y_pred)

    print(f"Normal Accuracy: {acc_normal:.4f}")

    # ---------------------------------------------------------
    # CONFUSION MATRIX (Normal)
    # ---------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title("Confusion Matrix - Solar Panel Fault Classification (Normal)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PROJECT_DIR}/{EXP_NAME}/confusion_matrix_normal.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # 🔥 YENİ: DETAILED CLASSIFICATION REPORT
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (NORMAL)")
    print("="*70)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Save reports
    with open(f"{PROJECT_DIR}/{EXP_NAME}/classification_report_normal.txt", "w") as f:
        f.write(report)

    # ---------------------------------------------------------
    # 🔥 YENİ: PER-CLASS F1 SCORE VISUALIZATION
    # ---------------------------------------------------------
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PROJECT_DIR}/{EXP_NAME}/per_class_metrics.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------
    # LOAD & DISPLAY TRAINING METRICS
    # ---------------------------------------------------------
    results_csv = f"{PROJECT_DIR}/{EXP_NAME}/results.csv"
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df = df.apply(pd.to_numeric, errors='ignore')
        
        print("\n[INFO] Training metrics summary:")
        print(df[['epoch', 'train/loss', 'val/loss', 'metrics/accuracy_top1']].tail(10))
        
        # 🔥 Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss', linewidth=2)
        if 'val/loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val/loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy curves 
        if 'metrics/accuracy_top1' in df.columns:
            axes[0, 1].plot(
                df['epoch'],
                df['metrics/accuracy_top1'],
                label='Top-1 Accuracy',
                linewidth=2
            )
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy Metrics')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # Learning rate
        if 'lr/pg0' in df.columns:
            axes[1, 0].plot(df['epoch'], df['lr/pg0'], linewidth=2, color='red')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(alpha=0.3)
        
        # Overfitting check
        if 'train/loss' in df.columns and 'val/loss' in df.columns:
            axes[1, 1].plot(df['epoch'], df['val/loss'] - df['train/loss'], linewidth=2, color='purple')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Loss - Train Loss')
            axes[1, 1].set_title('Overfitting Indicator (closer to 0 is better)')
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{PROJECT_DIR}/{EXP_NAME}/training_curves.png", dpi=300)
        plt.show()

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Best Model Path: {best_model_path}")
    print(f"Top-1 Accuracy (Normal): {acc_normal:.4f}")
    print(f"Total Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"All outputs saved in: {PROJECT_DIR}/{EXP_NAME}")
    print("="*70)


# ---------------------------------------------------------
# 🔥 ÖNEMLİ: Windows multiprocessing hatası için gerekli
# ---------------------------------------------------------
if __name__ == '__main__':
    main()