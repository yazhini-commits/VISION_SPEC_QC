# Dataset Data Card – VisionSpec QC

## Dataset Overview
This dataset is designed for visual quality control of industrial products,
focusing on binary classification between non-defective (good) and defective samples.

## Source
- Primary dataset: Bottle inspection images
- Supplementary data: Curated clean samples for non-defective class

## Classes
- Good: Non-defective product images
- Defective: Images containing visible defects

## Data Splits
- Training: 70%
- Validation: 15%
- Testing: 15%

## Preprocessing
- Images resized to 224×224 pixels
- Pixel values normalized to [0,1]

## Augmentation (Training Only)
- Horizontal flipping
- Rotation
- Brightness and contrast variation
- Affine scaling

## Bias & Limitations
- Dataset primarily focuses on surface-level defects
- Performance may vary on unseen defect categories

## Intended Use
- Academic research
- Industrial proof-of-concept for automated quality inspection
