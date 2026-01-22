import os
import cv2
import numpy as np
import pandas as pd
import re

# ==============================================================
# 1. FINAL PATH MAPPING
# ==============================================================
BASE = r"c:\Users\shasm\Downloads\VISION_SPEC_QC-main\VISION_SPEC_QC-main"

# The specific folder you found!
TEST_DEFECTIVE_DIR = os.path.join(BASE, "processed_data", "test", "defective")
HEATMAP_DIR = os.path.join(BASE, "gradcam_outputs", "final_heatmap_output")

OUTPUT_DIR = os.path.join(BASE, "gradcam_outputs", "final_test_localization")
RESULT_CSV = os.path.join(BASE, "gradcam_outputs", "test_localization_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# 2. MATCHING LOGIC
# ==============================================================
def find_test_image(number_str, folder):
    if not os.path.exists(folder): return None
    for f in os.listdir(folder):
        # Matches (46) inside the filename
        if f"({number_str})" in f and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(folder, f)
    return None

# ==============================================================
# 3. PROCESSING
# ==============================================================
print(f"Targeting TEST images in: {TEST_DEFECTIVE_DIR}")
heatmaps = [f for f in os.listdir(HEATMAP_DIR) if f.lower().endswith('.png')]
results = []

for h_name in heatmaps:
    match = re.search(r'\((\d+)\)', h_name)
    if match:
        num_id = match.group(1)
        test_img_path = find_test_image(num_id, TEST_DEFECTIVE_DIR)
        
        if test_img_path:
            h_img = cv2.imread(os.path.join(HEATMAP_DIR, h_name))
            test_img = cv2.imread(test_img_path)
            
            # Standardize sizes
            h_img = cv2.resize(h_img, (512, 512))
            test_img = cv2.resize(test_img, (512, 512))
            
            # --- IoU CALCULATION ---
            h_gray = cv2.cvtColor(h_img, cv2.COLOR_BGR2GRAY) / 255.0
            pred = h_gray > 0.5 # AI's focus area
            
            t_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) / 255.0
            # Detects dark defects (scratches/cracks) against lighter bottle
            truth = t_gray < 0.7 
            
            intersection = np.logical_and(pred, truth).sum()
            union = np.logical_or(pred, truth).sum()
            iou = intersection / union if union > 0 else 0
            
            # --- VISUALIZATION ---
            # Blue tint = AI Guess | Red tint = Actual Defect
            overlay = test_img.copy()
            overlay[pred] = [255, 0, 0]   # Blue (Prediction)
            overlay[truth] = [0, 0, 255]  # Red (Ground Truth)
            
            final_view = cv2.addWeighted(test_img, 0.6, overlay, 0.4, 0)
            cv2.imwrite(os.path.join(OUTPUT_DIR, h_name), final_view)
            
            print(f" TEST MATCH: ID {num_id} | IoU: {iou:.4f}")
            results.append({"image": h_name, "iou": iou})
        else:
            print(f" ID {num_id} not found in test/defective.")

# Save CSV
df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"\n✔ Done! Final Test Avg IoU: {df['iou'].mean() if not df.empty else 0:.4f}")