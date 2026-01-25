## Dataset Preparation 

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

## Image Preprocessing 

- Resized all images to 224×224 pixels
- Normalized pixel values to range [0, 1]
- Stored preprocessed images in a separate directory
- Ensured consistency across training, validation, and testing data

## Data Augmentation

- Applied augmentation techniques on training data only
- Techniques used:
  - Horizontal flipping
  - Rotation
  - Brightness and contrast adjustment
  - Zoom scaling
- Augmented images stored separately to avoid data leakage
- Improved dataset diversity and model robustness

## Dataset Validation & Visualization 

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

## Baseline Model Training 

- Implemented a baseline CNN classifier using ResNet18
- Trained on preprocessed dataset with standardized transforms
- Validated performance using a held-out validation set
- Saved trained baseline model for future comparison
### Confidence & Failure Analysis
- Implemented confidence-aware inference
- Added human-review fallback for low confidence cases
- Introduced error categorization with focus on false negatives
- Improved system safety and trustworthiness
## Explainability & Model Interpretation

- Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize model decision regions.
- Enabled spatial interpretation of predictions to understand why an image was classified as good or defective.
- Focused on improving transparency and trust in the quality-control system.
- Ensured explainability was added without affecting inference performance.

### Grad-CAM Integration

- Used the final convolutional layer of the ResNet18 backbone for Grad-CAM generation.
- Computed gradient-based importance weights to highlight discriminative regions.
- Generated heatmaps showing areas that most influenced the model’s prediction.
- Overlaid heatmaps on original images for intuitive visualization.

### Output Management

- Saved Grad-CAM visualizations in a dedicated `gradcam_outputs/` directory.
- Preserved original image resolution with superimposed heatmaps.
- Outputs can be used for inspection, debugging, and demonstration purposes.

### Impact on Quality Control

- Improved interpretability of model predictions for human reviewers.
- Helped verify that the model focuses on actual defect regions instead of background noise.
- Supported auditability and acceptance in industrial inspection workflows.

### Observations

- Defective samples showed strong activation around damaged or irregular surface areas.
- Good samples generally produced low-intensity or distributed activations.
- Grad-CAM assisted in understanding uncertain predictions and model limitations.
## Explainable AI & Visual Debugging 

- Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to explain model predictions
- Visualized important regions influencing predictions for GOOD and DEFECTIVE samples
- Generated heatmaps overlaid on original bottle images
- Improved transparency and trust in AI-based quality inspection

### Grad-CAM Integration

- Integrated Grad-CAM with the final convolutional layer (layer4) of ResNet18
- Generated class-specific activation maps during inference
- Normalized activation maps and applied color mapping for better visualization
- Stored Grad-CAM outputs for inspection review and auditing

### Model Debugging & Analysis

- Used Grad-CAM to analyze incorrect and uncertain predictions
- Identified background bias and overfitting to bottle edges in some cases
- Verified whether the model focuses on actual defect regions
- Helped diagnose misclassification scenarios effectively

### Dashboard Integration

- Embedded Grad-CAM visual explanations into the Streamlit dashboard
- Displayed prediction result, confidence score, severity level, and heatmap together
- Enabled human-in-the-loop validation for critical quality decisions

### Key Outcomes

- Enhanced explainability without affecting inference performance
- Improved model debugging and interpretability
- Achieved Explainable AI (XAI) compliance for industrial visual inspection

### Limitations

- Grad-CAM highlights approximate regions, not exact defect boundaries
- Heatmaps may appear diffused for subtle or low-contrast defects

### Future Enhancements

- Combine Grad-CAM with defect localization techniques
- Improve attention sharpness using advanced explainability methods
- Extend explainability support for multiple defect categories
