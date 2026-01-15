import os
import cv2
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

orig_dir = os.path.join(PROJECT_ROOT, "processed_data", "train", "good")
aug_dir = os.path.join(PROJECT_ROOT, "augmented_data", "train", "good")

orig_img = cv2.imread(os.path.join(orig_dir, os.listdir(orig_dir)[0]))
aug_img = cv2.imread(os.path.join(aug_dir, os.listdir(aug_dir)[0]))

orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(orig_img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Augmented Image")
plt.imshow(aug_img)
plt.axis("off")

plt.show()
