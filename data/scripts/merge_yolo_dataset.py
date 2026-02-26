"""
Merge a new YOLO dataset into the main detection_yolo dataset.
Handles class mapping and file copying.
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Set


def read_classes(classes_file: Path) -> List[str]:
    """Read classes from a classes.txt file."""
    if not classes_file.exists():
        return []
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()]


def remap_label_file(src_label: Path, class_mapping: Dict[int, int], dst_label: Path) -> None:
    """Remap class IDs in a YOLO label file."""
    if not src_label.exists():
        return
    
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    
    lines = src_label.read_text(encoding="utf-8").strip().split("\n")
    remapped_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        old_class_id = int(parts[0])
        if old_class_id in class_mapping:
            new_class_id = class_mapping[old_class_id]
            remapped_line = f"{new_class_id} {' '.join(parts[1:])}\n"
            remapped_lines.append(remapped_line)
    
    if remapped_lines:
        dst_label.write_text("".join(remapped_lines), encoding="utf-8")


def copy_image_with_unique_name(src_img: Path, dst_dir: Path, prefix: str = "") -> Path:
    """Copy image to destination with a unique name if needed."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Add prefix to avoid name conflicts
    if prefix:
        new_name = f"{prefix}_{src_img.name}"
    else:
        new_name = src_img.name
    
    dst_path = dst_dir / new_name
    
    # If file exists, add a counter
    counter = 1
    while dst_path.exists():
        stem = src_img.stem
        suffix = src_img.suffix
        new_name = f"{prefix}_{stem}_{counter}{suffix}" if prefix else f"{stem}_{counter}{suffix}"
        dst_path = dst_dir / new_name
        counter += 1
    
    shutil.copy2(src_img, dst_path)
    return dst_path


def merge_datasets(
    main_dataset_root: Path,
    new_dataset_root: Path,
    new_dataset_classes: List[str],
    new_to_main_class_mapping: Dict[str, int],
    main_classes: List[str],
) -> None:
    """Merge new dataset into main dataset."""
    
    print(f"Main dataset has {len(main_classes)} classes")
    print(f"New dataset has {len(new_dataset_classes)} classes: {new_dataset_classes}")
    
    # Create class mapping for label remapping
    # new_dataset_classes: ['3', 'bird_drop', 'snow_covered'] -> indices 0, 1, 2
    label_class_mapping: Dict[int, int] = {}
    for new_idx, new_class in enumerate(new_dataset_classes):
        if new_class in new_to_main_class_mapping:
            main_class_id = new_to_main_class_mapping[new_class]
            label_class_mapping[new_idx] = main_class_id
            print(f"  Mapping '{new_class}' (ID {new_idx} in new) -> '{main_classes[main_class_id]}' (ID {main_class_id} in main)")
        else:
            print(f"  Warning: Class '{new_class}' not found in mapping, skipping")
    
    # Process each split
    splits = ["train", "valid", "test"]
    split_mapping = {"valid": "val"}  # Map valid -> val for main dataset
    
    total_images = 0
    total_labels = 0
    
    for split in splits:
        new_split_dir = new_dataset_root / split
        if not new_split_dir.exists():
            print(f"  Warning: Split '{split}' not found in new dataset, skipping")
            continue
        
        new_images_dir = new_split_dir / "images"
        new_labels_dir = new_split_dir / "labels"
        
        # Determine main dataset split name
        main_split = split_mapping.get(split, split)
        main_images_dir = main_dataset_root / "images" / main_split
        main_labels_dir = main_dataset_root / "labels" / main_split
        
        if not new_images_dir.exists() or not new_labels_dir.exists():
            print(f"  Warning: Images or labels directory missing for split '{split}', skipping")
            continue
        
        # Copy images and remap labels
        image_files = list(new_images_dir.glob("*.*"))
        image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
        
        print(f"\nProcessing split '{split}' -> '{main_split}':")
        print(f"  Found {len(image_files)} images")
        
        copied = 0
        for img_file in image_files:
            # Copy image
            dst_img = copy_image_with_unique_name(img_file, main_images_dir, prefix="roboflow")
            copied += 1
            
            # Remap and copy label
            label_file = new_labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = main_labels_dir / f"{dst_img.stem}.txt"
                remap_label_file(label_file, label_class_mapping, dst_label)
                total_labels += 1
            else:
                # Create empty label file if image has no annotations
                dst_label = main_labels_dir / f"{dst_img.stem}.txt"
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.write_text("", encoding="utf-8")
        
        total_images += copied
        print(f"  Copied {copied} images and labels")
    
    print(f"\nTotal: {total_images} images, {total_labels} labels merged")
    
    # Update classes.txt (should already be correct, but verify)
    main_classes_file.write_text("\n".join(main_classes), encoding="utf-8")
    
    # Update data.yaml
    data_yaml = (
        f"path: {main_dataset_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: {main_classes}\n"
    )
    (main_dataset_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    
    print(f"\nDataset merge complete!")
    print(f"Updated classes: {len(main_classes)}")
    print(f"Main dataset location: {main_dataset_root}")


def main():
    parser = argparse.ArgumentParser(description="Merge a new YOLO dataset into the main detection_yolo dataset")
    parser.add_argument(
        "--new-dataset",
        type=str,
        default=r"ilkel dataset do not use for traning\Solar Panel Fault detection.v1i.yolov12",
        help="Path to the new dataset directory",
    )
    parser.add_argument(
        "--main-dataset",
        type=str,
        default="data/detection_yolo",
        help="Path to the main dataset directory",
    )
    args = parser.parse_args()
    
    main_dataset_root = Path(args.main_dataset)
    new_dataset_root = Path(args.new_dataset)
    
    if not main_dataset_root.exists():
        raise SystemExit(f"Main dataset not found: {main_dataset_root}")
    if not new_dataset_root.exists():
        raise SystemExit(f"New dataset not found: {new_dataset_root}")
    
    # Read new dataset classes from data.yaml
    new_data_yaml = new_dataset_root / "data.yaml"
    if not new_data_yaml.exists():
        raise SystemExit(f"data.yaml not found in new dataset: {new_data_yaml}")
    
    # Parse new dataset classes
    yaml_content = new_data_yaml.read_text(encoding="utf-8")
    # Extract names from yaml (simple parsing)
    names_line = [line for line in yaml_content.split("\n") if "names:" in line]
    if not names_line:
        raise SystemExit("Could not find 'names:' in new dataset data.yaml")
    
    # Parse: names: ['3', 'bird_drop', 'snow_covered']
    names_str = names_line[0].split("names:")[1].strip()
    # Remove brackets and quotes
    names_str = names_str.strip("[]")
    new_dataset_classes = [name.strip().strip("'\"") for name in names_str.split(",")]
    
    print(f"New dataset classes: {new_dataset_classes}")
    
    # Read main dataset classes
    main_classes = read_classes(main_dataset_root / "classes.txt")
    if not main_classes:
        # Try reading from data.yaml
        main_data_yaml = main_dataset_root / "data.yaml"
        if main_data_yaml.exists():
            yaml_content = main_data_yaml.read_text(encoding="utf-8")
            names_line = [line for line in yaml_content.split("\n") if "names:" in line]
            if names_line:
                names_str = names_line[0].split("names:")[1].strip()
                names_str = names_str.strip("[]")
                main_classes = [name.strip().strip("'\"") for name in names_str.split(",")]
    
    if not main_classes:
        raise SystemExit("Could not read main dataset classes")
    
    print(f"Main dataset classes ({len(main_classes)}): {main_classes}")
    
    # Create class mapping: map new dataset classes to main dataset class IDs
    # Add new classes to main if they don't exist
    new_to_main_class_mapping: Dict[str, int] = {}
    
    for new_class in new_dataset_classes:
        if new_class in main_classes:
            # Class already exists, use existing ID
            new_to_main_class_mapping[new_class] = main_classes.index(new_class)
        else:
            # New class, add to main classes
            # Handle special case: '3' might be a placeholder or unnamed class
            if new_class == '3':
                # Rename to something more descriptive
                new_class_name = 'unknown_defect'
                if new_class_name not in main_classes:
                    main_classes.append(new_class_name)
                    new_to_main_class_mapping[new_class] = len(main_classes) - 1
                    print(f"  Mapped class '3' to '{new_class_name}' (ID {new_to_main_class_mapping[new_class]})")
                else:
                    new_to_main_class_mapping[new_class] = main_classes.index(new_class_name)
                continue
            
            # Add new class to main classes
            main_classes.append(new_class)
            new_to_main_class_mapping[new_class] = len(main_classes) - 1
            print(f"  Added new class '{new_class}' to main dataset (ID {new_to_main_class_mapping[new_class]})")
    
    # Merge datasets
    merge_datasets(main_dataset_root, new_dataset_root, new_dataset_classes, new_to_main_class_mapping, main_classes)


if __name__ == "__main__":
    main()

