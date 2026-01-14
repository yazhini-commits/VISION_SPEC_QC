import os
import cv2
import albumentations as A

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

INPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "augmented_data", "train")

classes = ["good", "defective"]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(scale=(0.8, 1.2), p=0.5)
])

for cls in classes:
    input_path = os.path.join(INPUT_DIR, cls)
    output_path = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(output_path, exist_ok=True)

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        augmented = transform(image=image)["image"]
        save_name = f"aug_{img_name}"
        cv2.imwrite(os.path.join(output_path, save_name), augmented)

print("Data augmentation completed successfully!")
