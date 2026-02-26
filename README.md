## Solar Panel AI

Multi-class solar panel fault classification and object detection using YOLO (Ultralytics) and related models.

This repo contains:
- **Training scripts** for classification and detection (YOLOv11/12/8, Faster R-CNN + EfficientNet AE, etc.)
- **Dataset conversion utilities** to turn raw datasets into YOLO-ready formats
- **Analysis utilities** such as Grad-CAM and metrics visualization

All large datasets and model weights are **kept locally** and excluded from Git via `.gitignore`, so the repo stays lightweight on GitHub.

---

### Project structure (high level)

- **`data/scripts/`**: main Python scripts  
  - `train_yolov12_classification.py` – YOLO multi-class fault classification  
  - `train_yolov12_object_detection.py` – YOLO-based object detection training  
  - `train_yolov8.py`, `train_classification.py`, `train_detection_frcnn_ae.py` – other training variants  
  - `resnet train code .py` – ResNet-50 classifier training using pure PyTorch / torchvision  
  - `convert_*_to_yolo*.py`, `merge_yolo_dataset.py`, `organize_dataset.py` – dataset conversion / organization helpers  
  - `gradcam_produce.py` – Grad-CAM visualization for trained classifiers  
- **`data/detection_yolo*/`**: YOLO detection datasets, configs, and helper scripts  
- **`data/classification/`**: classification datasets (EL / RGB) – **ignored in Git**  
- **`data/models/`**: model package and local weight files (`.pt` are ignored in Git)  
- **`runs/`**: training outputs, logs, and visualizations – **ignored in Git**  
- **`ilkel dataset do not use for traning/` + other raw datasets**: legacy/raw data used during experimentation – **ignored in Git**

You can keep your full datasets in these folders locally; Git will not try to upload them.

---

### Environment & installation

- **Python**: recommended 3.10–3.11
- **Dependencies**: listed in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

This will pull in PyTorch, Ultralytics, NumPy/Pandas, visualization libs, and OpenCV, matching what the scripts use.

If you plan to use YOLOv12 bleeding-edge features, you can optionally run:

```bash
python data/scripts/install_yolov12_from_github.py
```

to install/upgrade Ultralytics from GitHub.

---

### Datasets (not stored in the repo)

Large image datasets and annotations **are not committed** to GitHub. Instead:

- Place your **classification datasets** under `data/classification/...`
- Place YOLO **detection datasets** under the provided `data/detection_yolo*/` folders following the YOLO directory structure
- Use the provided conversion scripts in `data/scripts/` (e.g. `convert_voc_to_yolo.py`, `convert_faulty_to_yolo_detection.py`, etc.) to transform raw datasets into YOLO format

The `.gitignore` is configured so you can freely keep all your local data in these folders without bloating the repository.

---

### Training examples

From the project root:

```bash
# Multi-class classification with YOLO
python data/scripts/train_yolov12_classification.py

# Object detection with YOLO
python data/scripts/train_yolov12_object_detection.py

# YOLOv8-based training
python data/scripts/train_yolov8.py

# ResNet-50 classification (torchvision)
python "data/scripts/resnet train code .py"

# Grad-CAM visualization for a trained classifier
python data/scripts/gradcam_produce.py
```

You can edit the config sections at the top of each script (paths, hyperparameters, model names) to match your local dataset and hardware.

---

### Pushing to GitHub

1. **Initialize Git** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial clean Solar Panel AI project"
   ```
2. Create a new GitHub repo and add it as a remote:
   ```bash
   git remote add origin https://github.com/<your-username>/solar-panel-ai.git
   git push -u origin main
   ```

Because of `.gitignore`, your large datasets, experiment outputs, and `.pt` weight files remain local, and only clean, reusable project code/configs are uploaded.

