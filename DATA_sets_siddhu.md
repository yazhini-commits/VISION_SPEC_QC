## Dataset Preparation – Day 1 (12 Jan 2026)

- Collected visual inspection images for quality control
- Classified images into:
  - Good (non-defective)
  - Defective samples
- Organized dataset into train, validation, and test folders
- Followed standard ML dataset structuring practices
### Dataset Source
- Dataset: MVTec Anomaly Detection Dataset
- Category used: Bottle
- Defect types merged into a single "defective" class
- Dataset split followed 70% train, 15% validation, 15% test
## Image Preprocessing – Day 2 (13 Jan 2026)

- Resized all images to 224×224 pixels
- Normalized pixel values to range [0, 1]
- Stored preprocessed images in a separate directory
- Ensured consistency across training, validation, and testing data
