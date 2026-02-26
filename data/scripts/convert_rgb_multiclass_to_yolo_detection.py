import argparse
import shutil
from pathlib import Path
from typing import List


def write_yolo_annotation(label_path: Path, class_id: int):
    """Write YOLO format annotation with full-image bounding box."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    # Full image bounding box: center at (0.5, 0.5), width=1.0, height=1.0
    with label_path.open("w", encoding="utf-8") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def convert_rgb_multiclass_to_detection(rgb_multiclass_root: Path, out_root: Path) -> None:
    """
    Convert RGB multiclass classification dataset to YOLO detection format.
    Preserves existing train/val/test splits.
    Each image gets a full-image bounding box labeled with its class.
    """
    # Class folders (same in train/val/test)
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
    
    classes = [class_name_map[cls] for cls in class_dirs]
    
    print(f"Classes: {classes}")
    print(f"Total classes: {len(classes)}")
    
    # Process each split (train/val/test)
    splits = ["train", "val", "test"]
    total_images = 0
    
    for split_name in splits:
        split_dir = rgb_multiclass_root / split_name
        if not split_dir.exists():
            print(f"Warning: Split folder not found: {split_dir}")
            continue
        
        print(f"\nProcessing {split_name} split...")
        split_count = 0
        
        for cls_dir in class_dirs:
            class_folder = split_dir / cls_dir
            if not class_folder.exists():
                print(f"  Warning: Class folder not found: {class_folder}")
                continue
            
            class_name = class_name_map[cls_dir]
            class_id = classes.index(class_name)
            
            # Find all image files in this class folder
            image_files = []
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                image_files.extend(list(class_folder.glob(f"*{ext}")))
            
            # Filter out system files
            image_files = [f for f in image_files if f.name.lower() not in {"desktop.ini", "thumbs.db"}]
            
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_path in image_files:
                # Copy image
                dst_img = out_root / "images" / split_name / img_path.name
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                
                # Write YOLO label with full-image bounding box
                dst_lbl = out_root / "labels" / split_name / (img_path.stem + ".txt")
                write_yolo_annotation(dst_lbl, class_id)
                split_count += 1
        
        print(f"  Total {split_name} images: {split_count}")
        total_images += split_count
    
    # Save class list and data.yaml
    (out_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    
    # Use relative path for data.yaml
    data_yaml = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names: {classes}\n"
    )
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    
    # Count images in each split
    train_count = len(list((out_root / "images" / "train").glob("*.*"))) if (out_root / "images" / "train").exists() else 0
    val_count = len(list((out_root / "images" / "val").glob("*.*"))) if (out_root / "images" / "val").exists() else 0
    test_count = len(list((out_root / "images" / "test").glob("*.*"))) if (out_root / "images" / "test").exists() else 0
    
    print(f"\n{'='*60}")
    print(f"YOLO detection dataset ready at: {out_root}")
    print(f"Classes: {len(classes)}")
    print(f"Total images: {total_images}")
    print(f"  - Train: {train_count}")
    print(f"  - Val: {val_count}")
    print(f"  - Test: {test_count}")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser(description="Convert RGB multiclass classification dataset to YOLO detection format")
    ap.add_argument("--rgb-multiclass-root", default="data/classification/rgb/multiclass")
    ap.add_argument("--out", default="data/detection_yolo_rgb_multiclass")
    args = ap.parse_args()
    
    rgb_multiclass_root = Path(args.rgb_multiclass_root)
    if not rgb_multiclass_root.exists():
        raise SystemExit(f"RGB multiclass folder not found: {rgb_multiclass_root}")
    
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    
    convert_rgb_multiclass_to_detection(rgb_multiclass_root, out_root)


if __name__ == "__main__":
    main()

