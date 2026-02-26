import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


RNG = random.Random(42)


def split_ids(ids: List[str], train_ratio=0.8, val_ratio=0.1) -> Dict[str, List[str]]:
    """Split image IDs into train/val/test sets."""
    RNG.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def write_yolo_annotation(label_path: Path, class_id: int):
    """Write YOLO format annotation with full-image bounding box."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    # Full image bounding box: center at (0.5, 0.5), width=1.0, height=1.0
    with label_path.open("w", encoding="utf-8") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def convert_classification_to_detection(faulty_root: Path, out_root: Path) -> None:
    """
    Convert Faulty_solar_panel classification dataset to YOLO detection format.
    Each image gets a full-image bounding box labeled with its class.
    """
    # Class folders in Faulty_solar_panel
    class_dirs = [
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
    ]
    
    # Map class folder names to clean class names
    class_name_map = {
        "Bird-drop": "bird_drop",
        "Clean": "clean",
        "Dusty": "dusty",
        "Electrical-damage": "electrical_damage",
        "Physical-Damage": "physical_damage",
        "Snow-Covered": "snow_covered",
    }
    
    classes = []
    all_items: List[Tuple[str, Path, str]] = []  # (unique_id, img_path, class_name)
    
    # Collect all images from each class folder
    for cls_dir in class_dirs:
        class_folder = faulty_root / cls_dir
        if not class_folder.exists():
            print(f"Warning: Class folder not found: {class_folder}")
            continue
        
        class_name = class_name_map.get(cls_dir, cls_dir.lower().replace("-", "_").replace(" ", "_"))
        if class_name not in classes:
            classes.append(class_name)
        
        class_id = classes.index(class_name)
        
        # Find all image files in this folder
        for img_path in class_folder.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}:
                continue
            if img_path.name.lower() in {"desktop.ini", "thumbs.db"}:
                continue
            
            # Create unique ID from class and filename
            unique_id = f"{class_name}_{img_path.stem}"
            all_items.append((unique_id, img_path, class_name))
    
    if not all_items:
        raise SystemExit("No images found in Faulty_solar_panel dataset!")
    
    print(f"Found {len(all_items)} images across {len(classes)} classes")
    print(f"Classes: {classes}")
    
    # Split into train/val/test
    splits = split_ids([uid for uid, _, _ in all_items], train_ratio=0.8, val_ratio=0.1)
    uid_to_item = {uid: (img_path, class_name) for uid, img_path, class_name in all_items}
    
    # Process each split
    for split_name, ids in splits.items():
        print(f"\nProcessing {split_name} split: {len(ids)} images")
        for uid in ids:
            img_path, class_name = uid_to_item[uid]
            class_id = classes.index(class_name)
            
            # Copy image
            dst_img = out_root / "images" / split_name / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            
            # Write YOLO label with full-image bounding box
            dst_lbl = out_root / "labels" / split_name / (img_path.stem + ".txt")
            write_yolo_annotation(dst_lbl, class_id)
    
    # Save class list and data.yaml
    (out_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    data_yaml = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names: {classes}\n"
    )
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    
    print(f"\n✓ YOLO detection dataset ready at: {out_root}")
    print(f"✓ Classes: {len(classes)}")
    print(f"✓ Total images: {len(all_items)}")
    print(f"  - Train: {len(splits['train'])}")
    print(f"  - Val: {len(splits['val'])}")
    print(f"  - Test: {len(splits['test'])}")


def main():
    ap = argparse.ArgumentParser(description="Convert Faulty_solar_panel classification dataset to YOLO detection format")
    ap.add_argument("--faulty-root", default="ilkel dataset do not use for traning/Faulty_solar_panel")
    ap.add_argument("--out", default="data/detection_yolo_faulty")
    args = ap.parse_args()
    
    faulty_root = Path(args.faulty_root)
    if not faulty_root.exists():
        raise SystemExit(f"Faulty_solar_panel folder not found: {faulty_root}")
    
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    
    convert_classification_to_detection(faulty_root, out_root)


if __name__ == "__main__":
    main()

