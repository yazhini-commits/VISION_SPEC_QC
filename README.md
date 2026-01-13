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

