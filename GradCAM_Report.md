# Grad-CAM Explainability Study and Implementation  
**Project:** Vision-Based Quality Control (VISION_SPEC_QC)  
**Task:** Explainability & Interpretability Engineer  
**Contributor:** Jasmine  
 

---

## Abstract
Convolutional Neural Networks (CNNs) achieve high performance in image classification tasks but often lack interpretability. Gradient-weighted Class Activation Mapping (Grad-CAM) is an Explainable AI (XAI) technique that provides visual explanations for CNN predictions.  
This report documents the study of Grad-CAM fundamentals (Day 1) and the design and implementation of a Grad-CAM-based explainability approach for a defect detection model (Day 2).

---

## Day 1: Study of Grad-CAM Fundamentals

### 1. Introduction
Deep learning models are frequently criticized for operating as “black boxes.” In critical applications such as industrial quality inspection, understanding *why* a model makes a prediction is as important as the prediction itself. Explainable AI (XAI) techniques help bridge this gap by improving transparency and trust.

Grad-CAM is a widely used XAI method designed specifically for CNN-based vision models.

---

### 2. What is Grad-CAM?
Grad-CAM (Gradient-weighted Class Activation Mapping) is a visualization technique that highlights image regions most influential in predicting a specific class. It uses the gradients of the target class flowing into the final convolutional layer to produce a localization heatmap.

---

### 3. Key Advantages of Grad-CAM
- Model-agnostic for CNN architectures  
- Class-discriminative explanations  
- Works without modifying model architecture  
- Enhances trust and interpretability  
- Useful for debugging and validation  

---

### 4. Working Principle
The Grad-CAM process consists of the following steps:
1. Forward pass of input image through the CNN  
2. Selection of target class score  
3. Gradient computation with respect to the last convolutional layer  
4. Global average pooling of gradients to obtain weights  
5. Weighted combination of feature maps  
6. ReLU activation to focus on positive influence  
7. Overlay of heatmap on the original image  

---

### 5. Applications
- Industrial defect detection  
- Medical image diagnosis  
- Autonomous systems  
- Surveillance and security  
- Quality assurance systems  

---

### 6. Day 1 Conclusion
Grad-CAM provides an effective and intuitive approach to understanding CNN decisions. A solid understanding of its fundamentals is essential before practical implementation in real-world applications.

---

## Day 2 – Design the Grad-CAM Approach

### Objectives
- Design a structured Grad-CAM explainability workflow
- Define gradient flow from prediction to convolutional layers

### Design Decisions
- Post-training explainability (no retraining)
- Use final convolutional layer for spatial relevance
- Class-specific gradient computation
- Heatmap normalization for visual clarity

### Planned Grad-CAM Flow
1. Load trained CNN model (`my_model.h5`)
2. Build a gradient model:
   - Input → last convolutional layer
   - Input → model predictions
3. Compute gradients using `tf.GradientTape`
4. Global average pooling of gradients
5. Weighted feature map aggregation
6. ReLU activation
7. Resize and overlay heatmap on input image

### Outcome
- Grad-CAM design finalized and ready for implementation
  
---

  ## Day 3 – Identify Target Layer for Grad-CAM

### Objective
To identify and validate the most suitable convolutional layer for Grad-CAM heatmap generation.

### Rationale
Grad-CAM requires a layer that:
- Preserves spatial information
- Is deep enough to capture semantic features
- Directly influences the final prediction

Therefore, the **last convolutional layer** of the CNN was selected.

### Implementation
- Programmatically inspected model layers
- Selected the final convolutional activation layer
- Verified output tensor dimensions

### Selected Target Layer
Layer Name: out_relu
Output Shape: (None, 7, 7, 1280)

---

##  Day 4 – Test Grad-CAM on Sample Images

### Objective
To test the Grad-CAM implementation on real dataset images and verify end-to-end heatmap generation.

### Dataset Used
- Source: `processed_data/test/defective`
- Image type: PCB defect images

### Implementation Steps
1. Load trained CNN model (`my_model.h5`)
2. Read sample image using OpenCV
3. Apply preprocessing:
   - Resize to 224 × 224
   - Normalize pixel values
4. Forward pass through Grad-CAM model
5. Compute gradients for predicted class
6. Generate Grad-CAM heatmap
7. Resize heatmap to original image size
8. Superimpose heatmap on input image
9. Save output image

### Output
- Grad-CAM heatmap successfully generated
- Output saved as `gradcam_result.jpg`
- Console output includes:
  - Predicted class index
  - Confidence score

---

##  Day 5 – Grad-CAM Validation & Explainability Analysis

### Objective
To validate the correctness, reliability, and interpretability of the Grad-CAM outputs generated from the defect detection model.

### Validation Focus Areas
- Correct gradient flow from prediction layer to the target convolutional layer
- Proper heatmap generation and normalization
- Spatial alignment between heatmap and original image
- Meaningful visual correspondence between highlighted regions and defect areas

### Validation Steps
1. Verified that gradients are computed with respect to the last convolutional layer (`out_relu`)
2. Confirmed non-zero gradients for the predicted class
3. Applied ReLU to focus on positively contributing features
4. Normalized heatmap values between 0 and 1
5. Resized heatmap to match input image dimensions
6. Visually inspected overlay alignment and focus regions

### Example Console Output
```text
Model Prediction Index: 0
Confidence: ~0.77

---

## Day 6 – Grad-CAM Integration with Trained Model

### Objective
Integrate Grad-CAM with the trained CNN model (`my_model.h5`) to generate visual explanations for test images and validate model attention.

### Focus Areas
- Wrap model for Grad-CAM  
- Preprocess images (224×224, normalize)  
- Compute and overlay heatmaps  
- Annotate predictions with actual vs predicted label and confidence  
- Save outputs in `gradcam_outputs/`

### Steps
1. Load trained model and create Grad-CAM wrapper  
2. Preprocess images and normalize  
3. Compute Grad-CAM using `tf.GradientTape`  
4. Resize heatmap and overlay on original image  
5. Annotate with actual vs predicted class and confidence  
6. Save annotated outputs  

### Conclusion

Grad-CAM provides visual explanations for model predictions, highlighting key regions and supporting interpretability and validation of the defect detection workflow.



