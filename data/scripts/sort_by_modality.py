import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required. Install with: pip install pillow") from exc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def iter_image_files(root_dirs: Iterable[Path]) -> Iterable[Path]:
    for root in root_dirs:
        for path in root.rglob("*"):
            if path.suffix in SUPPORTED_EXTENSIONS and path.is_file():
                yield path


def detect_modality(image_path: Path) -> str:
    """Return 'el' for grayscale-like (single-channel) images, else 'rgb'."""
    with Image.open(image_path) as img:
        mode = img.mode
        # Common grayscale-like modes: L (8-bit), I;16 (16-bit), I (32-bit), F (32-bit float)
        if mode in {"1", "L", "I;16", "I", "F"}:
            return "el"
        # Some datasets store grayscale as RGB where all channels are equal. Check a small sample.
        if mode in {"RGB", "RGBA"}:
            try:
                sample = img.convert("RGB").resize((16, 16))
                bands = sample.split()
                # If channels are identical across a small sample, treat as EL
                if all(bands[0].tobytes() == b.tobytes() for b in bands[1:]):
                    return "el"
            except Exception:
                pass
            return "rgb"
        # Fallback: treat unknown modes as rgb
        return "rgb"


def write_registry(rows: List[Tuple[str, str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "source_root", "modality"])  # modality in {el, rgb}
        writer.writerows(rows)


def mirror_to_destination(rows: List[Tuple[str, str, str]], dest_root: Path) -> None:
    """Copy files to data/image_modality/{el,rgb}/... preserving relative paths after source root.

    On Windows, symlinks often require elevation; copying is safer and predictable.
    """
    for abs_path, source_root, modality in rows:
        src = Path(abs_path)
        # Compute relative path under its source root to preserve sub-structure
        try:
            rel = Path(os.path.relpath(src, Path(source_root)))
        except ValueError:
            # If relpath fails (e.g., on different drives), just flatten to filename
            rel = src.name
        target = dest_root / modality / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(src, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify images by modality: EL (grayscale) vs RGB.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "elpv-dataset/elpv_dataset/data/images",
            "PVELAD/EL2021/trainval/JPEGImages",
            "PVELAD/EL2021/test/JPEGImages",
            "PV-Multi-Defect-main/JPEGImages",
            "Faulty_solar_panel",
        ],
        help="Root directories to scan for images.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/registry/image_modality.csv",
        help="Path to write the image modality registry CSV.",
    )
    parser.add_argument(
        "--copy-to",
        default="",
        help="If set, copy images into this root under {el,rgb}/ preserving relative structure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p).resolve() for p in args.roots]
    rows: List[Tuple[str, str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for img_path in iter_image_files([root]):
            try:
                modality = detect_modality(img_path)
            except Exception:
                # If an image is unreadable, skip it but keep going
                continue
            rows.append((str(img_path), str(root), modality))

    write_registry(rows, Path(args.out_csv))

    if args.copy_to:
        mirror_to_destination(rows, Path(args.copy_to))

    print(f"Indexed {len(rows)} images. Registry -> {args.out_csv}")
    if args.copy_to:
        print(f"Copied into {args.copy_to} (subfolders: el/, rgb/)")


if __name__ == "__main__":
    main()


