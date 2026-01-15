import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DATA_DIRS = {
    "Train": os.path.join(PROJECT_ROOT, "processed_data", "train"),
    "Validation": os.path.join(PROJECT_ROOT, "processed_data", "val"),
    "Test": os.path.join(PROJECT_ROOT, "processed_data", "test"),
}

classes = ["good", "defective"]

print("DATASET SUMMARY\n")

for split, path in DATA_DIRS.items():
    print(f"{split}:")
    for cls in classes:
        cls_path = os.path.join(path, cls)
        count = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
        print(f"  {cls}: {count} images")
    print("-" * 30)
