import argparse
import json
import os
import sys
import shutil
from pathlib import Path
import urllib.request
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from ultralytics import YOLO
import threading
import time

def check_hardware_for_yolov12():
    """
    YOLOv12 için donanım uygunluğunu ve FlashAttention desteğini kontrol eder.
    """
    print("\n=== Donanım ve Ortam Kontrolü ===")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        major_ver = capability[0]
        
        print(f"✅ GPU Algılandı: {device_name} (Compute Capability: {major_ver}.{capability[1]})")
        
        if major_ver >= 8:
            print("🚀 FlashAttention Desteği: VAR (Ampere/Hopper mimarisi algılandı).")
            print("   YOLOv12 tam performansla çalışacaktır.")
        else:
            print("⚠️ FlashAttention Desteği: YOK veya SINIRLI (Eski mimari).")
            print("   Eğitim devam eder ancak 'Area Attention' modülleri daha yavaş çalışabilir.")
    else:
        print("❌ GPU Algılanmadı! Eğitim CPU üzerinde çok yavaş olacaktır.")
    print("="*40 + "\n")

def try_load_yolov12_from_github(model_name: str, models_dir: Path):
    """YOLOv12 modelini GitHub release'lerinden veya yerel yoldan yüklemeye çalışır."""
    github_paths = [
        f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}",
        f"https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/{model_name}", # Orijinal repo path'i eklendi
    ]
    
    # 1. Önce models klasöründe kontrol et
    models_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = models_dir / model_name
    
    if local_model_path.exists():
        try:
            print(f"Yerel model bulundu: {local_model_path}")
            model = YOLO(str(local_model_path))
            return model
        except Exception as e:
            print(f"Yerel model yüklenemedi: {e}")
    
    # 2. Ultralytics'in varsayılan yolu ile dene (otomatik indirme)
    try:
        print(f"'{model_name}' Ultralytics'ten yükleniyor...")
        model = YOLO(model_name)
        # Eğer başarılı olduysa, modeli models klasörüne kopyala
        if hasattr(model, 'ckpt_path') and Path(model.ckpt_path).exists():
            shutil.copy2(model.ckpt_path, local_model_path)
            print(f"Model models klasörüne kopyalandı: {local_model_path}")
        return model
    except Exception as e:
        print(f"Ultralytics yükleme başarısız oldu: {e}")
    
    # 3. GitHub'dan indirip models klasörüne kaydet
    if not local_model_path.exists():
        for url in github_paths:
            try:
                print(f"GitHub'dan indiriliyor: {url}")
                print(f"Hedef: {local_model_path}")
                urllib.request.urlretrieve(url, local_model_path)
                if local_model_path.exists():
                    print(f"Başarıyla indirildi: {local_model_path}")
                    break
            except Exception as e:
                print(f"İndirme başarısız ({url}): {e}")
                continue
    
    # 4. İndirilen modeli yükle
    if local_model_path.exists():
        try:
            model = YOLO(str(local_model_path))
            print(f"GitHub'dan indirilen model yüklendi: {model_name}")
            return model
        except Exception as e:
            print(f"Model dosyası bozuk veya uyumsuz: {e}")

    return None

def main():
    ap = argparse.ArgumentParser(description="YOLOv12/v11 Gelişmiş Eğitim Scripti")
    ap.add_argument("--data", default="data/detection_yolo_rgb_multiclass/data.yaml")
    ap.add_argument("--model", default="yolov12n.pt", help="Model: yolov12n/s/m/l/x veya yolov11 varyasyonları")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/yolov12")
    ap.add_argument("--name", default="rgb_detection")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    ap.add_argument("--lr0", type=float, default=0.01)
    # Augmentasyon argümanlarını opsiyonel olarak tutuyoruz, ancak varsayılan olarak modelin config'ine bırakacağız.
    args = ap.parse_args()

    # 1. Donanım Kontrolü
    check_hardware_for_yolov12()

    # 2. Model Yükleme (Fallback Mekanizması ile)
    # Models klasörünü belirle
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model = None
    if "yolov12" in args.model.lower():
        print(f"YOLOv12 yüklenmeye çalışılıyor: {args.model}")
        model = try_load_yolov12_from_github(args.model, models_dir)
    
    if model is None:
        # Önce models klasöründe kontrol et
        local_model_path = models_dir / args.model
        if local_model_path.exists():
            try:
                print(f"Models klasöründen yükleniyor: {local_model_path}")
                model = YOLO(str(local_model_path))
            except Exception as e:
                print(f"Models klasöründen yükleme başarısız: {e}")
        
        # Hala yüklenemediyse, Ultralytics standart yükleme dene
        if model is None:
            try:
                print(f"Ultralytics standart yükleme deneniyor: {args.model}")
                model = YOLO(args.model)
                # Başarılı olduysa models klasörüne kopyala
                if hasattr(model, 'ckpt_path') and Path(model.ckpt_path).exists():
                    shutil.copy2(model.ckpt_path, local_model_path)
                    print(f"Model models klasörüne kopyalandı: {local_model_path}")
            except Exception as e:
                print(f"HATA: {args.model} yüklenemedi. ({e})")
                # Fallback: models klasöründen yolo11n.pt dene
                fallback_path = models_dir / "yolo11n.pt"
                if fallback_path.exists():
                    print(f"Fallback: {fallback_path} kullanılıyor...")
                    model = YOLO(str(fallback_path))
                else:
                    print("Fallback: YOLOv11n.pt Ultralytics'ten yükleniyor...")
                    model = YOLO("yolo11n.pt")

    # 3. Eğitim
    print(f"\n{'='*80}")
    print(f"YOLOv12 Training: {args.name}".center(80))
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Image Size: {args.img_size}")
    print(f"Dataset: {args.data}")
    print(f"{'='*80}\n")
    
    # Function to print clean epoch summary
    def print_epoch_summary(results_file, last_epoch_shown):
        """Print clean summary of latest epoch"""
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                if len(df) > last_epoch_shown:
                    latest = df.iloc[-1]
                    current_epoch = int(latest['epoch'])
                    
                    # Calculate F1 score
                    precision = latest['metrics/precision(B)']
                    recall = latest['metrics/recall(B)']
                    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
                    
                    print(f"\n{'─'*80}")
                    print(f"Epoch {current_epoch:3d}/{args.epochs:3d} Summary:")
                    print(f"{'─'*80}")
                    print(f"  mAP50:    {latest['metrics/mAP50(B)']:.4f} ({latest['metrics/mAP50(B)']*100:.2f}%)")
                    print(f"  mAP50-95: {latest['metrics/mAP50-95(B)']:.4f} ({latest['metrics/mAP50-95(B)']*100:.2f}%)")
                    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
                    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
                    print(f"  F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
                    print(f"  Train Loss: {latest['train/box_loss']:.4f} (box) + {latest['train/cls_loss']:.4f} (cls) + {latest['train/dfl_loss']:.4f} (dfl)")
                    print(f"{'─'*80}\n")
                    return current_epoch
            except:
                pass
        return last_epoch_shown
    
    # Start training with cleaner output
    results_file = Path(args.project) / args.name / "results.csv"
    last_epoch = 0
    
    print("Starting training... (Progress will be shown below)\n")
    print("Note: Detailed batch progress is shown, clean epoch summaries will appear after each validation.\n")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        half=True,           # FP16 optimizasyonu
        workers=8,           # Worker sayısını artırdım
        patience=args.patience,
        lr0=args.lr0,
        exist_ok=True,       # Üzerine yazmaya izin ver
        plots=True,
        verbose=True,        # Show progress but we add clean summaries
        # Default augmentasyonları kullanmak daha güvenlidir.
        # İstersen buraya **vars(args) diyerek hepsini yine paslayabilirsin.
    )
    
    # Print final summary after each epoch completes
    # (This runs after training, but we can also check during training)
    if results_file.exists():
        last_epoch = print_epoch_summary(results_file, last_epoch)
    
    # Print training completion summary
    print(f"\n{'='*80}")
    print("Training Completed!".center(80))
    print(f"{'='*80}")

    # Print final training summary from results.csv
    results_csv = Path(args.project) / args.name / "results.csv"
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            if len(df) > 0:
                latest = df.iloc[-1]
                best_map50_idx = df['metrics/mAP50(B)'].idxmax()
                best_map50 = df.loc[best_map50_idx]
                
                # Calculate F1 score
                precision = latest['metrics/precision(B)']
                recall = latest['metrics/recall(B)']
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                best_precision = best_map50['metrics/precision(B)']
                best_recall = best_map50['metrics/recall(B)']
                best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall + 1e-6)
                
                print(f"\n{'─'*80}")
                print("Final Training Summary:")
                print(f"{'─'*80}")
                print(f"Epochs Completed: {len(df)}/{args.epochs}")
                print(f"Best mAP50: {best_map50['metrics/mAP50(B)']:.4f} ({best_map50['metrics/mAP50(B)']*100:.2f}%) at Epoch {int(best_map50['epoch'])}")
                print(f"Latest mAP50: {latest['metrics/mAP50(B)']:.4f} ({latest['metrics/mAP50(B)']*100:.2f}%)")
                print(f"Latest mAP50-95: {latest['metrics/mAP50-95(B)']:.4f} ({latest['metrics/mAP50-95(B)']*100:.2f}%)")
                print(f"Latest Precision: {precision:.4f} ({precision*100:.2f}%)")
                print(f"Latest Recall: {recall:.4f} ({recall*100:.2f}%)")
                print(f"Latest F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
                print(f"Best F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%) at Epoch {int(best_map50['epoch'])}")
                print(f"{'─'*80}\n")
        except Exception as e:
            print(f"Could not read training summary: {e}\n")
    
    # 4. Değerlendirme (Evaluation)
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    metrics_dir = Path(args.project) / args.name / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    if best_model_path.exists():
        print(f"\n=== En iyi model yükleniyor: {best_model_path} ===")
        model = YOLO(best_model_path)
        
        # Test seti üzerinde doğrulama
        print("\n=== Running comprehensive evaluation on test set ===")
        metrics = model.val(
            data=args.data,
            split="test",      # data.yaml içinde 'test' yoksa otomatik 'val' kullanır
            imgsz=args.img_size,
            batch=args.batch,
            device=args.device,
            half=True,
            save_json=True,   # Save JSON results
            save_hybrid=True, # Save hybrid labels
            plots=True       # Generate plots
        )
        
        # Get class names
        class_names = list(model.names.values()) if hasattr(model, 'names') else [f"class_{i}" for i in range(len(metrics.box.maps) if hasattr(metrics.box, 'maps') else 0)]
        
        # Print all metrics
        print("\n" + "="*80)
        print("=== COMPREHENSIVE TEST METRICS ===")
        print("="*80)
        
        # Overall metrics
        overall_f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6)
        print(f"\n{'Overall Metrics':<30} {'Value':<15}")
        print("-" * 50)
        print(f"{'mAP50 (IoU=0.50)':<30} {metrics.box.map50:.4f}")
        print(f"{'mAP50-95 (IoU=0.50:0.95)':<30} {metrics.box.map:.4f}")
        print(f"{'Precision':<30} {metrics.box.mp:.4f}")
        print(f"{'Recall':<30} {metrics.box.mr:.4f}")
        print(f"{'F1-Score':<30} {overall_f1:.4f}")
        
        # Per-class mAP metrics
        if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
            print(f"\n{'Per-Class mAP Metrics':<30}")
            print("-" * 80)
            print(f"{'Class':<25} {'mAP50':<12} {'mAP50-95':<12}")
            print("-" * 80)
            for i, (name, map_val) in enumerate(zip(class_names, metrics.box.maps)):
                map50_val = metrics.box.map50 if not hasattr(metrics.box, 'maps50') else (metrics.box.maps50[i] if hasattr(metrics.box, 'maps50') and len(metrics.box.maps50) > i else metrics.box.map50)
                print(f"{name:<25} {map50_val:<12.4f} {map_val:<12.4f}")
        
        # Save metrics to JSON
        metrics_dict = {
            "overall": {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "f1": float(overall_f1),
            },
            "per_class": {}
        }
        
        # Add per-class mAP metrics
        if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
            for i, (name, map_val) in enumerate(zip(class_names, metrics.box.maps)):
                map50_val = metrics.box.map50 if not hasattr(metrics.box, 'maps50') else (metrics.box.maps50[i] if hasattr(metrics.box, 'maps50') and len(metrics.box.maps50) > i else metrics.box.map50)
                metrics_dict["per_class"][name] = {
                    "mAP50": float(map50_val),
                    "mAP50-95": float(map_val)
                }
        
        # Save to JSON (initial save)
        with open(metrics_dir / "test_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save results.csv if available
        results_csv = Path(args.project) / args.name / "results.csv"
        if results_csv.exists():
            results_df = pd.read_csv(results_csv)
            results_df.to_json(metrics_dir / "training_results.json", orient="records", indent=2)
            print(f"\n✓ Training results saved to: {metrics_dir / 'training_results.json'}")
        
        # Extract and save confusion matrix with detailed metrics
        print("\n=== Extracting Confusion Matrix and Detailed Metrics ===")
        try:
            # Get confusion matrix from metrics
            if hasattr(metrics, 'confusion_matrix'):
                cm = metrics.confusion_matrix.matrix
                class_names = list(model.names.values()) if hasattr(model, 'names') else [f"class_{i}" for i in range(len(cm))]
                
                # Save confusion matrix as JSON
                cm_dict = {
                    "matrix": cm.tolist() if hasattr(cm, 'tolist') else cm,
                    "class_names": class_names
                }
                with open(metrics_dir / "confusion_matrix.json", "w") as f:
                    json.dump(cm_dict, f, indent=2)
                print(f"✓ Confusion matrix saved to: {metrics_dir / 'confusion_matrix.json'}")
                
                # Calculate detailed per-class metrics from confusion matrix
                cm_array = np.array(cm)
                total_samples = cm_array.sum()
                total_correct = cm_array.trace()
                overall_accuracy = total_correct / (total_samples + 1e-6)
                
                print(f"\n{'Overall Accuracy (from CM)':<30} {overall_accuracy:.4f}")
                metrics_dict["overall"]["accuracy"] = float(overall_accuracy)
                
                per_class_metrics = {}
                print(f"\n{'Detailed Per-Class Metrics':<30}")
                print("-" * 100)
                print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Support':<10}")
                print("-" * 100)
                
                for i, class_name in enumerate(class_names):
                    tp = float(cm_array[i, i])
                    fp = float(cm_array[:, i].sum() - tp)
                    fn = float(cm_array[i, :].sum() - tp)
                    tn = float(total_samples - tp - fp - fn)
                    
                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
                    support = int(tp + fn)  # Actual number of samples for this class
                    
                    per_class_metrics[class_name] = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "accuracy": float(accuracy),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                        "support": int(support)
                    }
                    
                    print(f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:<12.4f} {int(tp):<8} {int(fp):<8} {int(fn):<8} {support:<10}")
                
                # Add per-class detailed metrics to metrics dict
                for class_name, class_data in per_class_metrics.items():
                    if class_name in metrics_dict["per_class"]:
                        metrics_dict["per_class"][class_name].update(class_data)
                    else:
                        metrics_dict["per_class"][class_name] = class_data
                
                # Calculate macro and weighted averages
                precisions = [v["precision"] for v in per_class_metrics.values()]
                recalls = [v["recall"] for v in per_class_metrics.values()]
                f1_scores = [v["f1"] for v in per_class_metrics.values()]
                supports = [v["support"] for v in per_class_metrics.values()]
                
                macro_precision = np.mean(precisions)
                macro_recall = np.mean(recalls)
                macro_f1 = np.mean(f1_scores)
                
                weighted_precision = np.average(precisions, weights=supports) if sum(supports) > 0 else 0
                weighted_recall = np.average(recalls, weights=supports) if sum(supports) > 0 else 0
                weighted_f1 = np.average(f1_scores, weights=supports) if sum(supports) > 0 else 0
                
                print(f"\n{'Averaged Metrics':<30}")
                print("-" * 80)
                print(f"{'Metric':<20} {'Macro Avg':<15} {'Weighted Avg':<15}")
                print("-" * 80)
                print(f"{'Precision':<20} {macro_precision:<15.4f} {weighted_precision:<15.4f}")
                print(f"{'Recall':<20} {macro_recall:<15.4f} {weighted_recall:<15.4f}")
                print(f"{'F1-Score':<20} {macro_f1:<15.4f} {weighted_f1:<15.4f}")
                
                metrics_dict["overall"]["macro_avg"] = {
                    "precision": float(macro_precision),
                    "recall": float(macro_recall),
                    "f1": float(macro_f1)
                }
                metrics_dict["overall"]["weighted_avg"] = {
                    "precision": float(weighted_precision),
                    "recall": float(weighted_recall),
                    "f1": float(weighted_f1)
                }
                
                # Save updated metrics with all detailed scores
                with open(metrics_dir / "test_metrics.json", "w") as f:
                    json.dump(metrics_dict, f, indent=2)
                print(f"\n✓ All detailed metrics saved to: {metrics_dir / 'test_metrics.json'}")
        except Exception as e:
            print(f"Warning: Could not extract confusion matrix: {e}")
            import traceback
            traceback.print_exc()

        
        # 5. Gelişmiş Tahmin Döngüsü (OPTIMİZE EDİLDİ ⚡)
        print("\n=== Görsel Tahminler (Predictions) Oluşturuluyor ===")
        
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Test veya Val yolunu bul
        base_path = Path(data_config.get("path", "."))
        test_images_dir = base_path / data_config.get("test", "images/test")
        if not test_images_dir.exists():
             test_images_dir = base_path / data_config.get("val", "images/val")
        
        if test_images_dir.exists():
            print(f"Kaynak klasör: {test_images_dir}")
            
            all_image_results = []
            
            # STREAM=TRUE ile bellek dostu ve hızlı tahmin
            # Manuel batch döngüsüne gerek yok, Ultralytics bunu içeride optimize eder.
            results_generator = model.predict(
                source=str(test_images_dir),
                imgsz=args.img_size,
                conf=0.25,
                iou=0.45,
                save=False,       # Görüntüleri diske kaydetmek istersen True yap
                stream=True,      # Generator döndürür, RAM şişmez!
                verbose=False
            )
            
            print("Tahminler işleniyor...")
            for i, result in enumerate(results_generator):
                path = Path(result.path)
                
                # CPU'ya taşı ve Numpy'a çevir
                boxes = result.boxes.cpu().numpy()
                
                img_result = {
                    "image_name": path.name,
                    "num_detections": len(boxes),
                    "predictions": []
                }
                
                if boxes.shape[0] > 0:
                    for box in boxes:
                        # xyxy, conf, cls
                        coords = box.xyxy[0]
                        conf = box.conf[0]
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]
                        
                        img_result["predictions"].append({
                            "class_name": class_name,
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(coords[0]), "y1": float(coords[1]),
                                "x2": float(coords[2]), "y2": float(coords[3])
                            }
                        })
                
                all_image_results.append(img_result)
                
                if (i+1) % 50 == 0:
                    print(f"  {i+1} görüntü işlendi...", end='\r')

            # Sonuçları Kaydet
            with open(metrics_dir / "per_image_results.json", "w") as f:
                json.dump(all_image_results, f, indent=2)
            
            print(f"\n✓ Toplam {len(all_image_results)} görsel analiz edildi.")
            print(f"✓ Sonuçlar: {metrics_dir / 'per_image_results.json'}")
            
        else:
            print(f"⚠️ Test görsel klasörü bulunamadı: {test_images_dir}")
        
        # Final summary
        print(f"\n{'='*80}")
        print("=== TRAINING AND EVALUATION COMPLETE ===")
        print(f"{'='*80}")
        print(f"✓ All metrics saved to: {metrics_dir}")
        print(f"✓ Model weights: {best_model_path}")
        print(f"✓ Plots and visualizations: {Path(args.project) / args.name}")
        print(f"✓ Training results: {metrics_dir / 'training_results.json'}")
        print(f"✓ Test metrics: {metrics_dir / 'test_metrics.json'}")
        print(f"✓ Confusion matrix: {metrics_dir / 'confusion_matrix.json'}")
        print(f"✓ Per-image results: {metrics_dir / 'per_image_results.json'}")
    else:
        print(f"❌ En iyi model dosyası bulunamadı: {best_model_path}")

if __name__ == "__main__":
    main()