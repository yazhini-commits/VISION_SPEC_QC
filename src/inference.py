import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from datetime import datetime

# --------------------------------------------------
# PATHS
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_resnet18_v1.pth")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.json")

os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --------------------------------------------------
# IMPORTANT: SAME TRANSFORMS AS TRAINING
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# CLASS ORDER (MUST MATCH TRAINING FOLDER ORDER)
# Folder order used during training:
# train/
# ├── defective (class 0)
# └── good      (class 1)
# --------------------------------------------------
CLASS_NAMES = ["DEFECTIVE", "GOOD"]

# --------------------------------------------------
# SEVERITY LOGIC
# --------------------------------------------------
def severity_level(defect_prob):
    if defect_prob < 0.4:
        return "LOW"
    elif defect_prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def log_prediction(image_path, output):
    entry = {
        "image": os.path.basename(image_path),
        **output,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

# --------------------------------------------------
# PREDICT (DEFECT-SENSITIVE LOGIC)
# --------------------------------------------------
def predict(image_path, confidence_threshold=0.65, defect_threshold=0.60):
    """
    DEFECT-SENSITIVE INFERENCE LOGIC

    Priority:
    1. If defect probability is high → DEFECTIVE
    2. Else if confidence is low → UNCERTAIN
    3. Else → GOOD
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    # Probabilities
    defect_prob = probs[0].item()  # class 0 = DEFECTIVE
    good_prob = probs[1].item()    # class 1 = GOOD

    confidence = max(defect_prob, good_prob)

    # ---------------- DECISION LOGIC ----------------
    if defect_prob >= defect_threshold:
        result = "DEFECTIVE"
        severity = severity_level(defect_prob)

    elif confidence < confidence_threshold:
        result = "UNCERTAIN – HUMAN REVIEW REQUIRED"
        severity = None

    else:
        result = "GOOD"
        severity = None

    output = {
        "result": result,
        "confidence": round(confidence, 3),
        "severity": severity,
        "data_drift_warning": False
    }

    log_prediction(image_path, output)
    return output

# --------------------------------------------------
# CLI TEST
# --------------------------------------------------
if __name__ == "__main__":
    print("Inference module ready ✔")
