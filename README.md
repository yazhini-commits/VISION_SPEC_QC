# VisionSpec QC – Automated Visual Quality Control (Computer Vision)

## Overview
VisionSpec QC is a production-grade computer vision system for automated visual inspection
of Printed Circuit Boards (PCBs). The system classifies PCB images as **Pass** or **Defect**
and localizes defect regions using explainable AI techniques.

## Business Use Case
Manual PCB inspection is slow and inconsistent in high-speed manufacturing lines.
VisionSpec QC automates defect detection to improve quality, reduce human error,
and enable real-time inspection.

## Key Features
- Automated PCB defect classification (Pass / Defect)
- Transfer learning with pre-trained CNN models
- Data augmentation for robustness
- Defect localization using Grad-CAM
- Real-time inference with OpenCV

## Tech Stack
Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Scikit-learn

## Project Structure
VisionSpec-QC-Automated-Visual-Inspection/
├── data/ (raw, processed, augmented)
├── notebooks/ (EDA)
├── src/ (training, Grad-CAM, inference)
├── models/
├── outputs/
├── requirements.txt
└── README.md

## Workflow
1. PCB image acquisition  
2. Preprocessing & augmentation  
3. Transfer learning model training  
4. Grad-CAM explainability  
5. Real-time inference  

## Execution Plan
- Week 1: Data preparation & augmentation  
- Week 2: Model training  
- Week 3: Explainability (Grad-CAM)  
- Week 4: Real-time inference & validation  

## Performance
Designed for production environments with real-time inference capability
(>10 frames per second).

## Author
Zaalima Development Internship  
Production AI Project – Computer Vision (Project 3)


## Model Training Summary (Member-2)

- CNN Architecture: MobileNetV2
- Transfer Learning with ImageNet weights
- Base layers frozen during initial training
- Optimizer: Adam (learning rate = 0.0001)
- Loss Function: Binary Crossentropy
- Input Size: 224x224x3
- Final model saved at: saved_models/my_model.h5

This trained model is provided for downstream explainability (Grad-CAM) integration.


## Model Handover 

- Final trained model: saved_models/my_model.h5
- Input shape: 224×224×3
- Status: Ready for Grad-CAM explainability and deployment testing
- Owner: Member-2 (Model Development)


## Model Readiness

- Final trained model verified and load-tested
- Input shape: 224×224×3
- Output: Binary (Pass / Defect)
- Status: Ready for Grad-CAM explainability and deployment testing





## Model Explanation 

- Transfer learning enabled faster convergence and better generalization
- Data augmentation improved robustness to real-world variations
- Final optimized model saved and verified for downstream use