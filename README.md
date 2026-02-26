## Solar Panel AI

Deep learning project for **automatic solar panel defect classification and detection** from EL and RGB images.

The goal is to detect faults such as **bird droppings, dust, electrical damage, physical damage, snow cover, and more**, using modern computer vision models.

This repo includes:
- **Classification models**: YOLO-based classifiers and a ResNet-50 classifier (PyTorch / torchvision)
- **Object detection models**: YOLO-based detectors and Faster R-CNN + EfficientNet-AE
- **Dataset utilities**: scripts to convert and organize different public/own datasets into YOLO format
- **Analysis tools**: metrics analysis, confusion matrices, Grad-CAM visualizations

All large datasets and model weights are **kept locally** and excluded from Git via `.gitignore`, so the repo stays lightweight on GitHub.

---

### Project structure (high level)

- **`data/scripts/`** – main Python scripts  
  - `train_yolov12_classification.py` – YOLO multi-class fault classification on solar panel images  
  - `train_yolov12_object_detection.py` – YOLO-based object detection training for solar panel defects  
  - `train_yolov8.py`, `train_classification.py`, `train_detection_frcnn_ae.py` – additional YOLO / Faster R-CNN + AE training variants  
  - `resnet train code .py` – ResNet-50 classifier training using pure PyTorch / torchvision  
  - `convert_*_to_yolo*.py`, `merge_yolo_dataset.py`, `organize_dataset.py` – dataset conversion / organization helpers (VOC → YOLO, custom datasets → YOLO)  
  - `gradcam_produce.py` – Grad-CAM visualization for trained classifiers  
  - `analyze_results.py` – helper script to analyze YOLO training metrics (`results.csv`)  
- **`data/detection_yolo*/`** – YOLO detection datasets, configs, and helper scripts  
- **`data/classification/`** – classification datasets (EL / RGB) – **ignored in Git**  
- **`data/models/`** – model package and local weight files (`.pt` are ignored in Git)  
- **`runs/`** – training outputs, logs, and visualizations – **ignored in Git**  
- **`ilkel dataset do not use for traning/` + other raw datasets** – legacy/raw data used during experimentation – **ignored in Git**

You can keep your full datasets in these folders locally; Git will not try to upload them.

---

### Environment & installation

- **Python**: recommended 3.10–3.11  
- **Dependencies**: listed in `requirements.txt`

Install all dependencies:

```bash
pip install -r requirements.txt
```

This pulls in:
- PyTorch + torchvision
- Ultralytics (YOLO)
- NumPy / Pandas / scikit-learn
- Matplotlib / Seaborn / OpenCV  
matching what the training and analysis scripts use.

If you plan to use YOLOv12 bleeding-edge features, you can optionally run:

```bash
python data/scripts/install_yolov12_from_github.py
```

to install/upgrade Ultralytics directly from GitHub.

---

### Datasets (not stored in the repo)

Large image datasets and annotations **are not committed** to GitHub. Instead, place them locally as follows:

- Put **classification datasets** under:  
  - `data/classification/el/...`  
  - `data/classification/rgb/...`
- Put YOLO **detection datasets** under:  
  - `data/detection_yolo/...`  
  - `data/detection_yolo_rgb_multiclass/...`
- Use the conversion scripts in `data/scripts/` (for example):  
  - `convert_voc_to_yolo.py` – VOC XML → YOLO text format  
  - `convert_faulty_to_yolo_detection.py` – Faulty_solar_panel style → YOLO  
  - `convert_rgb_multiclass_to_yolo_detection.py`, `merge_yolo_dataset.py`, etc.

The `.gitignore` is configured so you can freely keep all your local data in these folders without bloating the repository.

---

### Training examples

From the project root:

```bash
# Multi-class classification with YOLO (Ultralytics)
python data/scripts/train_yolov12_classification.py

# Object detection with YOLO
python data/scripts/train_yolov12_object_detection.py

# YOLOv8-based training
python data/scripts/train_yolov8.py

# ResNet-50 classification (torchvision)
python "data/scripts/resnet train code .py"

# Grad-CAM visualization for a trained classifier
python data/scripts/gradcam_produce.py

# Analyze YOLO training results (mAP, loss curves, stability, etc.)
python analyze_results.py
```

Edit the config sections at the top of each script (paths, hyperparameters, model names) to match your local dataset layout and hardware.

---

### Typical workflow

1. **Prepare datasets**  
   - Organize your EL / RGB images into the expected directory structure.  
   - Use the conversion scripts to turn external datasets into YOLO format if needed.
2. **Install environment**  
   - `pip install -r requirements.txt`
3. **Train a model**  
   - For YOLO classification or detection, run the corresponding `train_*.py` script.  
   - For ResNet-50 classification, run `"data/scripts/resnet train code .py"`.
4. **Inspect results**  
   - Check `runs/` for Ultralytics logs, plots, and best checkpoints (local only).  
   - Optionally run `analyze_results.py` and `gradcam_produce.py` for deeper analysis.

Because of `.gitignore`, your large datasets, experiment outputs, and `.pt` weight files remain local, and only clean, reusable project code/configs are uploaded.
