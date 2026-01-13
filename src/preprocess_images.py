import os
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

BASE_DIR = os.path.join(PROJECT_ROOT, "data_set")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data")

IMG_SIZE = 224

splits = ["train", "val", "test"]
classes = ["good", "defective"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in splits:
    for cls in classes:
        input_path = os.path.join(BASE_DIR, split, cls)
        output_path = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(output_path, exist_ok=True)

        for img_name in os.listdir(input_path):
            img_path = os.path.join(input_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            save_path = os.path.join(output_path, img_name)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))

print("Image preprocessing completed successfully!")
