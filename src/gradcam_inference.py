import os
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

from gradcam_utils import GradCAM

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_resnet18.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# GRADCAM SETUP
# --------------------------------------------------
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# --------------------------------------------------
# RUN GRADCAM
# --------------------------------------------------
def run_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    cv2.imwrite(save_path, overlay)

    print(f"Grad-CAM saved to: {save_path}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    test_image = os.path.join(
    PROJECT_ROOT,
    "data_set",
    "test",
    "good",
    "good_00 (1).png"
)


    if os.path.exists(test_image):
        run_gradcam(test_image)
    else:
        print("No test image found for Grad-CAM.")
