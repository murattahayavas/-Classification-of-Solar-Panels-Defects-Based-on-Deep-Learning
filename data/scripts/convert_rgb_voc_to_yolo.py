import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


RNG = random.Random(42)


def parse_voc(xml_path: Path) -> Tuple[Optional[Tuple[int, int]], List[Tuple[str, float, float, float, float]]]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, []
    size = root.find("size")
    if size is None:
        return None, []
    w = int(size.findtext("width", default="0"))
    h = int(size.findtext("height", default="0"))
    boxes: List[Tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not name or bb is None:
            continue
        xmin = float(bb.findtext("xmin", default="0"))
        ymin = float(bb.findtext("ymin", default="0"))
        xmax = float(bb.findtext("xmax", default="0"))
        ymax = float(bb.findtext("ymax", default="0"))
        # Convert to YOLO: x_center, y_center, width, height normalized
        x_c = ((xmin + xmax) / 2.0) / max(1, w)
        y_c = ((ymin + ymax) / 2.0) / max(1, h)
        bw = (xmax - xmin) / max(1, w)
        bh = (ymax - ymin) / max(1, h)
        boxes.append((name, x_c, y_c, bw, bh))
    return (w, h), boxes


def gather_classes(voc_roots: List[Tuple[Path, Path]]) -> List[str]:
    classes: Set[str] = set()
    for img_dir, ann_dir in voc_roots:
        if not ann_dir.exists():
            continue
        for xml in ann_dir.glob("*.xml"):
            _, boxes = parse_voc(xml)
            for name, *_ in boxes:
                classes.add(name)
    return sorted(classes)


def split_ids(ids: List[str], train_ratio=0.8, val_ratio=0.1) -> Dict[str, List[str]]:
    RNG.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def write_yolo_annotation(label_path: Path, boxes: List[Tuple[int, float, float, float, float]]):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, x, y, w, h in boxes:
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def convert_and_copy(voc_roots: List[Tuple[Path, Path]], classes: List[str], out_root: Path) -> None:
    cls_to_id: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    # Build a global list of unique image ids to split by dataset and file stem
    all_items: List[Tuple[str, Path, Path]] = []  # (unique_id, img_path, xml_path)
    for img_dir, ann_dir in voc_roots:
        for xml in ann_dir.glob("*.xml"):
            stem = xml.stem
            img = None
            for ext in [".jpg", ".JPG", ".png", ".jpeg", ".PNG", ".JPEG"]:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            all_items.append((f"{ann_dir.name}_{stem}", img, xml))

    splits = split_ids([uid for uid, _, _ in all_items], train_ratio=0.8, val_ratio=0.1)
    uid_to_item = {uid: (img, xml) for uid, img, xml in all_items}

    for split_name, ids in splits.items():
        for uid in ids:
            img, xml = uid_to_item[uid]
            size, boxes = parse_voc(xml)
            yolo_boxes: List[Tuple[int, float, float, float, float]] = []
            for name, x, y, w, h in boxes:
                if name not in cls_to_id:
                    continue
                yolo_boxes.append((cls_to_id[name], x, y, w, h))

            # Copy image
            dst_img = out_root / "images" / split_name / img.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(img, dst_img)

            # Write label (can be empty for negative examples)
            dst_lbl = out_root / "labels" / split_name / (img.stem + ".txt")
            write_yolo_annotation(dst_lbl, yolo_boxes)

    # Save class list and data.yaml
    (out_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    data_yaml = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names: {classes}\n"
    )
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Convert PV-Multi-Defect (RGB) VOC to YOLO format for YOLOv12.")
    ap.add_argument("--pvmd-img", default="PV-Multi-Defect-main/JPEGImages")
    ap.add_argument("--pvmd-ann", default="PV-Multi-Defect-main/Annotations")
    ap.add_argument("--out", default="data/detection_yolo_rgb")
    args = ap.parse_args()

    voc_roots = [
        (Path(args.pvmd_img), Path(args.pvmd_ann)),
    ]

    classes = gather_classes(voc_roots)
    if not classes:
        raise SystemExit("No classes found in VOC annotations.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    convert_and_copy(voc_roots, classes, out_root)
    print(f"RGB YOLO dataset ready at: {out_root} | classes: {len(classes)}")
    print(f"Classes: {classes}")


if __name__ == "__main__":
    main()

