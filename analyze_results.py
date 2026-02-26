import pandas as pd
import numpy as np

df = pd.read_csv('runs/yolov12/rgb_multiclass_nano_v1/results.csv')

print("=" * 80)
print("TRAINING RESULTS ANALYSIS - YOLOv12 Nano")
print("=" * 80)
print(f"\nTotal Epochs Completed: {len(df)} / 120")
print(f"Training Time: {df['time'].iloc[-1]:.1f} seconds ({df['time'].iloc[-1]/60:.1f} minutes)")

print("\n" + "=" * 80)
print("BEST PERFORMANCE")
print("=" * 80)
best_map50_idx = df['metrics/mAP50(B)'].idxmax()
best_map50_95_idx = df['metrics/mAP50-95(B)'].idxmax()

print(f"\nBest mAP50: {df.loc[best_map50_idx, 'metrics/mAP50(B)']:.4f} ({df.loc[best_map50_idx, 'metrics/mAP50(B)']*100:.2f}%)")
print(f"  Achieved at Epoch: {int(df.loc[best_map50_idx, 'epoch'])}")
print(f"  Precision: {df.loc[best_map50_idx, 'metrics/precision(B)']:.4f}")
print(f"  Recall: {df.loc[best_map50_idx, 'metrics/recall(B)']:.4f}")

print(f"\nBest mAP50-95: {df.loc[best_map50_95_idx, 'metrics/mAP50-95(B)']:.4f} ({df.loc[best_map50_95_idx, 'metrics/mAP50-95(B)']*100:.2f}%)")
print(f"  Achieved at Epoch: {int(df.loc[best_map50_95_idx, 'epoch'])}")

print("\n" + "=" * 80)
print("LATEST EPOCH (81)")
print("=" * 80)
latest = df.iloc[-1]
print(f"mAP50:        {latest['metrics/mAP50(B)']:.4f} ({latest['metrics/mAP50(B)']*100:.2f}%)")
print(f"mAP50-95:     {latest['metrics/mAP50-95(B)']:.4f} ({latest['metrics/mAP50-95(B)']*100:.2f}%)")
print(f"Precision:    {latest['metrics/precision(B)']:.4f} ({latest['metrics/precision(B)']*100:.2f}%)")
print(f"Recall:       {latest['metrics/recall(B)']:.4f} ({latest['metrics/recall(B)']*100:.2f}%)")

print("\n" + "=" * 80)
print("TRAINING PROGRESS")
print("=" * 80)
first10 = df.head(10)
last10 = df.tail(10)

print(f"\nFirst 10 Epochs (Average):")
print(f"  mAP50:    {first10['metrics/mAP50(B)'].mean():.4f}")
print(f"  mAP50-95: {first10['metrics/mAP50-95(B)'].mean():.4f}")

print(f"\nLast 10 Epochs (Average):")
print(f"  mAP50:    {last10['metrics/mAP50(B)'].mean():.4f}")
print(f"  mAP50-95: {last10['metrics/mAP50-95(B)'].mean():.4f}")

improvement_map50 = ((last10['metrics/mAP50(B)'].mean() - first10['metrics/mAP50(B)'].mean()) / first10['metrics/mAP50(B)'].mean()) * 100
improvement_map50_95 = ((last10['metrics/mAP50-95(B)'].mean() - first10['metrics/mAP50-95(B)'].mean()) / first10['metrics/mAP50-95(B)'].mean()) * 100

print(f"\nImprovement:")
print(f"  mAP50:    +{improvement_map50:.1f}%")
print(f"  mAP50-95: +{improvement_map50_95:.1f}%")

print("\n" + "=" * 80)
print("LOSS TRENDS")
print("=" * 80)
print(f"Train Box Loss:   {df['train/box_loss'].iloc[0]:.4f} -> {df['train/box_loss'].iloc[-1]:.4f} ({((df['train/box_loss'].iloc[0] - df['train/box_loss'].iloc[-1]) / df['train/box_loss'].iloc[0] * 100):.1f}% decrease)")
print(f"Train Class Loss:  {df['train/cls_loss'].iloc[0]:.4f} -> {df['train/cls_loss'].iloc[-1]:.4f} ({((df['train/cls_loss'].iloc[0] - df['train/cls_loss'].iloc[-1]) / df['train/cls_loss'].iloc[0] * 100):.1f}% decrease)")
print(f"Val Box Loss:      {df['val/box_loss'].iloc[0]:.4f} -> {df['val/box_loss'].iloc[-1]:.4f} ({((df['val/box_loss'].iloc[0] - df['val/box_loss'].iloc[-1]) / df['val/box_loss'].iloc[0] * 100):.1f}% decrease)")

print("\n" + "=" * 80)
print("STABILITY ANALYSIS")
print("=" * 80)
last20 = df.tail(20)
map50_std = last20['metrics/mAP50(B)'].std()
map50_95_std = last20['metrics/mAP50-95(B)'].std()

print(f"Last 20 Epochs Stability:")
print(f"  mAP50 std dev:    {map50_std:.4f} (lower is better)")
print(f"  mAP50-95 std dev: {map50_95_std:.4f} (lower is better)")

if map50_std < 0.02:
    stability = "Very Stable"
elif map50_std < 0.04:
    stability = "Stable"
else:
    stability = "Somewhat Volatile"

print(f"\nOverall Assessment: {stability}")

print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
if latest['metrics/mAP50(B)'] > 0.85:
    grade = "EXCELLENT"
elif latest['metrics/mAP50(B)'] > 0.75:
    grade = "VERY GOOD"
elif latest['metrics/mAP50(B)'] > 0.65:
    grade = "GOOD"
else:
    grade = "NEEDS IMPROVEMENT"

print(f"\nCurrent Performance Grade: {grade}")
print(f"\nThe model is performing {'very well' if latest['metrics/mAP50(B)'] > 0.80 else 'well' if latest['metrics/mAP50(B)'] > 0.70 else 'moderately'} for a nano model!")
print("=" * 80)

