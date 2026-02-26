import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import time
import copy
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8

DATA_DIR = r"C:\Users\halil\Desktop\solar_panel_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_output_dir():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_dir = os.path.join(desktop, "ResNet_Sunum_Cikti")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

OUTPUT_DIR = get_output_dir()

def plot_history(history, save_path):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    val_f1 = history.get('val_f1', [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(18, 6))
    plt.style.use('ggplot')

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.xlabel('Epochs')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.xlabel('Epochs')

    plt.subplot(1, 3, 3)
    if val_f1:
        plt.plot(epochs_range, val_f1, label='Val F1 Score', color='purple')
        plt.legend(loc='lower right')
    plt.title('F1 Score')
    plt.xlabel('Epochs')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sunum_basari_grafigi.png'), dpi=300)

def plot_confusion_matrix(model, loader, classes, save_path):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'sunum_confusion_matrix.png'), dpi=300)

def main():
    torch.backends.cudnn.benchmark = True

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    class_names = train_dataset.classes
    num_classes = len(class_names)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    scaler = GradScaler()

    best_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }

    log_data = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= dataset_sizes['val']
        val_acc = val_corrects.double() / dataset_sizes['val']

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='weighted',
            zero_division=0
        )

        scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        history['val_f1'].append(f1)

        log_data.append({
            "Epoch": epoch + 1,
            "Train_Loss": epoch_loss,
            "Train_Acc": epoch_acc.item(),
            "Val_Loss": val_loss,
            "Val_Acc": val_acc.item(),
            "Val_F1": f1,
            "Val_Precision": precision,
            "Val_Recall": recall
        })

        pd.DataFrame(log_data).to_csv(
            os.path.join(OUTPUT_DIR, 'egitim_verileri.csv'),
            index=False
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, 'best_resnet50_presentation.pth')
            )

    plot_history(history, OUTPUT_DIR)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_resnet50_presentation.pth')))
    plot_confusion_matrix(model, val_loader, class_names, OUTPUT_DIR)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
