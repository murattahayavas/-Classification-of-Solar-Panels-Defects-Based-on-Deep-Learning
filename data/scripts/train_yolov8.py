import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 on prepared YOLO dataset (data.yaml)")
    ap.add_argument("--data", default="data/detection_yolo/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="Base model: yolov8n.pt/yolov8s.pt/yolov8m.pt ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/yolov8")
    ap.add_argument("--name", default="el_detection")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = ap.parse_args()

    # Model yükle
    model = YOLO(args.model)

    # Eğitim
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=0,          # GPU 0 (RTX 4050)
        half=True,         # FP16 ile hız ve VRAM optimizasyonu
        workers=4,         # DataLoader hızını artırır
        patience=args.patience,  # Early stopping
    )

    # En iyi ağırlıklarla test/val doğrulaması
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    model = YOLO(best_model_path)
    metrics = model.val(
        data=args.data,
        split="test",      # "test" yoksa "val" olarak değiştir
        imgsz=args.img_size,
        batch=args.batch,
        device=0,
        half=True
    )

    print(metrics)

if __name__ == "__main__":
    main()
