"""
Helper script to install YOLOv12 from GitHub if available.
Run this before training to ensure YOLOv12 is available.
"""
import subprocess
import sys
from pathlib import Path


def install_from_github():
    """Try to install YOLOv12 from GitHub repositories."""
    repos = [
        "https://github.com/ultralytics/ultralytics.git",  # Main ultralytics repo
        "https://github.com/ultralytics/assets.git",  # Assets repo
    ]
    
    print("Attempting to install/update ultralytics from GitHub...")
    try:
        # Update ultralytics to latest from GitHub
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/ultralytics/ultralytics.git", 
            "--upgrade"
        ])
        print("✓ Successfully updated ultralytics from GitHub")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install from GitHub: {e}")
        print("Trying to install latest from PyPI instead...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "ultralytics", "--upgrade"
            ])
            print("✓ Updated ultralytics from PyPI")
            return True
        except Exception as e2:
            print(f"✗ Failed: {e2}")
            return False


if __name__ == "__main__":
    success = install_from_github()
    if success:
        print("\n✓ Installation complete. You can now try training with YOLOv12.")
        print("Example: python -m data.scripts.train_yolov12 --model yolov12n.pt")
    else:
        print("\n✗ Installation failed. Using YOLOv11/YOLOv8 as fallback.")

