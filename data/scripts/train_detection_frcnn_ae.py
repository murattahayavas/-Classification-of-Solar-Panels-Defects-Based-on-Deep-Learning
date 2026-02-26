import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import yaml

from data.models.fasterrcnn_efficientnet_ae import FasterRCNNWithAE


class YoloTxtDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, img_size: int = 640):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.ids = [p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        self.img_size = img_size
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            p = self.images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(stem)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_t = self.tf(img)

        # Labels
        boxes = []
        labels = []
        lbl_path = self.labels_dir / f"{stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) != 5 and len(parts) != 6:
                    continue
                cls_id = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])
                # Convert YOLO (normalized cx,cy,w,h) to absolute xyxy at original size
                cx, cy = x * w, y * h
                bw_abs, bh_abs = bw * w, bh * h
                xmin = max(0.0, cx - bw_abs / 2)
                ymin = max(0.0, cy - bh_abs / 2)
                xmax = min(w, cx + bw_abs / 2)
                ymax = min(h, cy + bh_abs / 2)
                # Scale to resized image
                sx, sy = self.img_size / w, self.img_size / h
                boxes.append([xmin * sx, ymin * sy, xmax * sx, ymax * sy])
                labels.append(cls_id)
        target: Dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def load_data_yaml(path: Path) -> Tuple[Path, Path, Path, List[str]]:
    y = yaml.safe_load(path.read_text())
    root = Path(y.get("path", path.parent.as_posix()))
    train = root / y["train"]
    val = root / y["val"]
    test = root / y.get("test", y["val"])
    names = y["names"]
    return train, val, test, names


def main():
    ap = argparse.ArgumentParser(description="Train Faster R-CNN (EfficientNet backbone) with AE auxiliary loss on YOLO-labeled data.")
    ap.add_argument("--data", default="data/detection_yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ae-weight", type=float, default=0.1)
    ap.add_argument("--out", default="runs/frcnn_ae")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    ap.set_defaults(amp=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img, val_img, test_img, names = load_data_yaml(Path(args.data))
    train_ds = YoloTxtDataset(train_img, Path(str(train_img).replace("images", "labels")), img_size=args.img_size)
    val_ds = YoloTxtDataset(val_img, Path(str(val_img).replace("images", "labels")), img_size=args.img_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = FasterRCNNWithAE(num_classes=len(names), ae_loss_weight=args.ae_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_dl:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                losses, loss_sum = model(images, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss_sum.item())
        total_loss /= max(1, len(train_dl))

        # Simple validation: forward pass loss on val set
        model.train()  # keep train mode for loss computation
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dl:
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                    losses, loss_sum = model(images, targets)
                val_loss += float(loss_sum.item())
        val_loss /= max(1, len(val_dl))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "classes": names,
                "backbone": "tf_efficientnet_b0",
                "img_size": args.img_size,
            }, out_dir / "best.pt")
        print(f"Epoch {epoch}/{args.epochs} | train_loss={total_loss:.4f} | val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()


