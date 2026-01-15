import os
import cv2
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

base_dir = os.path.join(PROJECT_ROOT, "processed_data", "train")

classes = ["good", "defective"]

plt.figure(figsize=(6,6))
idx = 1

for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    images = os.listdir(cls_path)[:3]

    for img in images:
        img_path = os.path.join(cls_path, img)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(2,3,idx)
        plt.imshow(image)
        plt.title(cls)
        plt.axis("off")
        idx += 1

plt.suptitle("Dataset Sample Visualization")
plt.show()
