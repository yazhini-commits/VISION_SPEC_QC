import os
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data", "train")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "train_mean.npy")

classes = ["good", "defective"]
means = []

for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)

    if not os.path.exists(cls_path):
        raise FileNotFoundError(f"Missing folder: {cls_path}")

    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (224, 224))
        means.append(image.mean())

train_mean = float(np.mean(means))

os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
np.save(OUTPUT_PATH, train_mean)

print("Training data statistics saved successfully")
print(f"Training mean pixel value: {train_mean:.2f}")
