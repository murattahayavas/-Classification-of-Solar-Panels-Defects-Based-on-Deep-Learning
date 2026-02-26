import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Ensure project root is on sys.path for package imports when running via file path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.models.asddnet import ASDDNet
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_t, eval_t


def build_dataloaders(data_root: Path, img_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_t, eval_t = build_transforms(img_size)
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_t)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_t)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_t)
    class_names = train_ds.classes

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl, class_names


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "asddnet":
        model = ASDDNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, List[int], List[int]]:
    model.eval()
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = correct / max(1, total)
    return acc, all_labels, all_preds


def train_one_epoch(model: nn.Module, dl: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dl, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / max(1, len(dl.dataset))


def save_metrics(out_dir: Path, split_name: str, class_names: List[str], y_true: List[int], y_pred: List[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    (out_dir / f"{split_name}_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"{split_name}_confusion_matrix.json").write_text(json.dumps(cm.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train image classification (ImageFolder) and report metrics.")
    ap.add_argument("--data-root", required=True, help="Folder with train/ val/ test/ subfolders")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "asddnet"])
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="runs/classification")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl, class_names = build_dataloaders(Path(args.data_root), args.img_size, args.batch_size, args.num_workers)
    model = build_model(args.arch, num_classes=len(class_names), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_acc, y_true_val, y_pred_val = evaluate(model, val_dl, device)
        save_metrics(out_dir, f"epoch{epoch:03d}_val", class_names, y_true_val, y_pred_val)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "arch": args.arch,
                "img_size": args.img_size,
            }, out_dir / "best.pt")
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    # Final evaluation on test
    test_acc, y_true_test, y_pred_test = evaluate(model, test_dl, device)
    save_metrics(out_dir, "test", class_names, y_true_test, y_pred_test)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()


