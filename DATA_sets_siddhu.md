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

## Data Augmentation – Day 3 (14 Jan 2026)

- Applied augmentation techniques on training data only
- Techniques used:
  - Horizontal flipping
  - Rotation
  - Brightness and contrast adjustment
  - Zoom scaling
- Augmented images stored separately to avoid data leakage
- Improved dataset diversity and model robustness

## Dataset Validation & Visualization – Day 4 (15 Jan 2026)

- Performed dataset sanity checks to validate class distribution
- Verified consistency across training, validation, and test splits
- Visualized original and augmented samples to confirm augmentation effectiveness
- Ensured no data leakage between dataset splits
- Finalized data pipeline for model training

## Known Failure Scenarios

- Extremely subtle defects with minimal texture variation
- Low-contrast defects under poor lighting conditions
- Defects partially occluded or outside the focal region

These cases are identified for future dataset expansion and model improvement.

## Data Integrity Validation

- Verified absence of corrupted or unreadable images
- Ensured no overlap between training, validation, and test sets
- Maintained strict split isolation to prevent data leakage

## Baseline Model Training – Day 5 (16 Jan 2026)

- Implemented a baseline CNN classifier using ResNet18
- Trained on preprocessed dataset with standardized transforms
- Validated performance using a held-out validation set
- Saved trained baseline model for future comparison
