import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


RNG = random.Random(42)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stratified_split(items: Sequence[Tuple[str, str]], ratios=(0.7, 0.15, 0.15)) -> Dict[str, List[Tuple[str, str]]]:
    """Stratify by label. items: list of (image_path, class_label)."""
    per_label: Dict[str, List[str]] = defaultdict(list)
    for p, label in items:
        per_label[label].append(p)
    splits: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for label, paths in per_label.items():
        RNG.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train = paths[:n_train]
        val = paths[n_train : n_train + n_val]
        test = paths[n_train + n_val :]
        splits["train"].extend((p, label) for p in train)
        splits["val"].extend((p, label) for p in val)
        splits["test"].extend((p, label) for p in test)
    return splits


def copy_split(splits: Dict[str, List[Tuple[str, str]]], out_root: Path) -> None:
    for split_name, rows in splits.items():
        for src, label in rows:
            dst = out_root / split_name / label / Path(src).name
            ensure_dir(dst.parent)
            if not dst.exists():
                shutil.copy2(src, dst)


def build_el_binary_from_elpv(elpv_root: Path, out_root: Path) -> None:
    """EL binary classification using ELPV labels.csv: 0.0->clean, 1.0->defective.

    Robustly resolves relative image paths based on the directory of labels.csv.
    """
    labels_csv = elpv_root / "elpv_dataset" / "data" / "labels.csv"
    if not labels_csv.exists():
        print(f"[EL-Binary] labels.csv not found at {labels_csv}")
        return
    items: List[Tuple[str, str]] = []
    base_dir = labels_csv.parent  # typically: elpv-dataset/elpv_dataset/data
    missing = 0
    with labels_csv.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_path, flag = parts[0], parts[1]
            # Primary: relative to labels.csv directory
            candidates = [
                base_dir / rel_path,
                elpv_root / rel_path,
                elpv_root / "elpv_dataset" / rel_path,
                elpv_root / "elpv_dataset" / "data" / rel_path,
            ]
            img_path = next((p for p in candidates if p.exists()), None)
            if img_path is None:
                missing += 1
                continue
            label = "defective" if flag.startswith("1") else "clean"
            items.append((str(img_path), label))
    if not items:
        print("[EL-Binary] No items found.")
        return
    splits = stratified_split(items)
    copy_split(splits, out_root)
    total = sum(len(v) for v in splits.values())
    print(f"[EL-Binary] Wrote: {total} files -> {out_root} (missing from CSV: {missing})")


def parse_voc_objects(xml_path: Path) -> List[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    names: List[str] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is not None and name_el.text:
            names.append(name_el.text.strip())
    return names


def dominant_label(labels: List[str]) -> str:
    if not labels:
        return "_background"
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def build_el_multiclass_from_voc(pvelad_root: Path, pvmd_root: Path, out_root: Path) -> None:
    """EL multiclass from VOC: use dominant object class per image.

    - PVELAD: PVELAD/EL2021/trainval/{JPEGImages,Annotations}
    - PV-MD:  PV-Multi-Defect-main/{JPEGImages,Annotations}
    """
    items: List[Tuple[str, str]] = []

    # PVELAD
    pva_img = pvelad_root / "EL2021" / "trainval" / "JPEGImages"
    pva_ann = pvelad_root / "EL2021" / "trainval" / "Annotations"
    if pva_img.exists() and pva_ann.exists():
        for xml in pva_ann.glob("*.xml"):
            labels = parse_voc_objects(xml)
            if not labels:
                continue
            label = dominant_label(labels)
            img = pva_img / (xml.stem + ".jpg")
            if img.exists():
                items.append((str(img), label))

    # PV-Multi-Defect
    pvmd_img = pvmd_root / "JPEGImages"
    pvmd_ann = pvmd_root / "Annotations"
    if pvmd_img.exists() and pvmd_ann.exists():
        for xml in pvmd_ann.glob("*.xml"):
            labels = parse_voc_objects(xml)
            if not labels:
                continue
            label = dominant_label(labels)
            img = pvmd_img / (xml.stem + ".jpg")
            if img.exists():
                items.append((str(img), label))

    if not items:
        print("[EL-Multiclass] No items found.")
        return
    splits = stratified_split(items)
    copy_split(splits, out_root)
    print(f"[EL-Multiclass] Wrote: {sum(len(v) for v in splits.values())} files -> {out_root}")


def build_rgb_multiclass_from_folders(faulty_root: Path, out_root: Path) -> None:
    """RGB multiclass from folder names under Faulty_solar_panel."""
    class_dirs = [
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
    ]
    items: List[Tuple[str, str]] = []
    for cls_dir in class_dirs:
        cdir = faulty_root / cls_dir
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file():
                items.append((str(p), cls_dir.replace(" ", "_")))
    if not items:
        print("[RGB-Multiclass] No items found.")
        return
    splits = stratified_split(items)
    copy_split(splits, out_root)
    print(f"[RGB-Multiclass] Wrote: {sum(len(v) for v in splits.values())} files -> {out_root}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build classification datasets (EL binary/multiclass, RGB multiclass) with splits.")
    ap.add_argument("--elpv-root", default="elpv-dataset", help="Root of ELPV dataset")
    ap.add_argument("--pvelad-root", default="PVELAD", help="Root of PVELAD dataset")
    ap.add_argument("--pvmd-root", default="PV-Multi-Defect-main", help="Root of PV-Multi-Defect dataset")
    ap.add_argument("--faulty-root", default="Faulty_solar_panel", help="Root of Faulty_solar_panel dataset")
    ap.add_argument("--out-root", default="data/classification", help="Output root for classification datasets")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)

    # EL binary (ELPV)
    build_el_binary_from_elpv(Path(args.elpv_root), out_root / "el" / "binary")

    # EL multiclass (PVELAD + PV-MD)
    build_el_multiclass_from_voc(Path(args.pvelad_root), Path(args.pvmd_root), out_root / "el" / "multiclass")

    # RGB multiclass (Faulty_solar_panel)
    build_rgb_multiclass_from_folders(Path(args.faulty_root), out_root / "rgb" / "multiclass")


if __name__ == "__main__":
    main()


