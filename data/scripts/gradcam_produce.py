import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "runs/classification/yolov8m-cls_v1/weights/best.pt"
IMAGE_PATH = "data/classification/rgb/test/physical_damage/Physical (41).jpg"
OUTPUT_DIR = "runs/classification/yolov8m-cls_v1/gradcam"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = YOLO(MODEL_PATH)
model.model.to(device)
model.model.eval()

# -------------------------------------------------
# 🔥 CORRECT TARGET LAYER (YOLOv8-CLS)
# -------------------------------------------------
target_layer = model.model.model[-3]  # LAST CONV BEFORE GAP
print("✅ Target layer:", target_layer)

# -------------------------------------------------
# HOOKS
# -------------------------------------------------
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1)
img_tensor = img_tensor.unsqueeze(0) / 255.0
img_tensor = img_tensor.to(device)
img_tensor.requires_grad_(True)

# -------------------------------------------------
# FORWARD
# -------------------------------------------------
outputs = model.model(img_tensor)
logits = outputs[0]

pred_class = logits.argmax(dim=1).item()
print(f"🎯 Predicted class index: {pred_class}")

# -------------------------------------------------
# BACKWARD
# -------------------------------------------------
model.model.zero_grad()
logits[0, pred_class].backward()

# -------------------------------------------------
# GRAD-CAM
# -------------------------------------------------
grads = gradients[0].detach().cpu().numpy()[0]
acts = activations[0].detach().cpu().numpy()[0]

weights = grads.mean(axis=(1, 2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

# -------------------------------------------------
# VISUALIZE
# -------------------------------------------------
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)

out_path = os.path.join(
    OUTPUT_DIR,
    f"gradcam_class_{pred_class}.jpg"
)

cv2.imwrite(out_path, overlay)
print(f"🔥 Grad-CAM saved to: {out_path}")
