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

## Day 2: Design and Implementation of Grad-CAM

### 1. Objective
To design and implement a Grad-CAM explainability pipeline for a trained CNN model used in visual defect detection.

---


At the time of implementation, test samples were available only for the defective class.

---

### 2. Model Details
- Framework: TensorFlow / Keras  
- Model file: `my_model.h5`  
- Input size: 224 × 224 × 3  
- Classification type: Binary (Defective / Non-defective)  

---

### 3. Grad-CAM Design Approach
The Grad-CAM pipeline was implemented using the following steps:

1. Load the trained `.h5` model  
2. Identify the last convolutional layer  
3. Preprocess input images  
4. Compute class-specific gradients using `GradientTape`  
5. Generate Grad-CAM heatmaps  
6. Overlay heatmaps on original images  
7. Save results for analysis  

---

### 4. Tools and Libraries Used
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Matplotlib  

---

### 5. Results
- Grad-CAM heatmaps successfully generated  
- Outputs saved for multiple defective test images  
- Heatmaps highlight relevant structural and surface regions  
- Model confidence values were consistently high  

---

### 6. Observations
- The model focuses on visually meaningful regions  
- Heatmaps align with expected defect-sensitive areas  
- Absence of non-defective test images limits comparative analysis  

---

### 7. Day 2 Conclusion
The Grad-CAM explainability approach was successfully designed and implemented. The generated visualizations confirm that the model relies on relevant image features, thereby improving transparency and reliability in defect detection.

---

## Overall Conclusion
This two-day task achieved both theoretical understanding and practical implementation of Grad-CAM. The study and results demonstrate the effectiveness of Grad-CAM as a reliable explainability tool for CNN-based quality inspection systems.

---

## Grad-CAM Explainability

Grad-CAM was applied to the trained MobileNetV2-based classifier to visualize
important regions influencing defect prediction.

The heatmaps clearly highlight solder joints and defect-prone areas,
confirming that the model focuses on meaningful PCB regions rather than background noise.

This improves trust and interpretability of the defect detection system.





