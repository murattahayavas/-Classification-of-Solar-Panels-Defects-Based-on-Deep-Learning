import os

BASE_PATH = r"C:\Users\Taha\Desktop\Solar Panel AI\data\classification\rgb"
SPLITS = ["train", "val", "test"]

EXPECTED_CLASSES = {
    "bird_drop": ["Bird-drop", "Bird_drop", "bird-drop"],
    "clean": ["Clean"],
    "dusty": ["Dusty"],
    "electrical_damage": ["Electrical-damage", "Electrical_Damage"],
    "physical_damage": ["Physical-Damage", "Physical_damage"],
    "snow_covered": ["Snow-Covered", "Snow_covered"]
}

def fix_folders():
    for split in SPLITS:
        split_path = os.path.join(BASE_PATH, split)
        if not os.path.exists(split_path):
            print(f"[WARN] {split_path} not found")
            continue

        print(f"\n[INFO] Checking {split} set")

        for correct_name, variants in EXPECTED_CLASSES.items():
            for var in variants:
                old_path = os.path.join(split_path, var)
                new_path = os.path.join(split_path, correct_name)

                if os.path.exists(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  ✔ Renamed: {var} → {correct_name}")

        print(f"[DONE] {split} checked")

if __name__ == "__main__":
    fix_folders()
