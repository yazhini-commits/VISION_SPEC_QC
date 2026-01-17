import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_resnet18.pth")
STATS_PATH = os.path.join(PROJECT_ROOT, "models", "train_mean.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if os.path.exists(STATS_PATH):
    train_mean = float(np.load(STATS_PATH))
else:
    train_mean = None

def severity_level(defect_prob):
    if defect_prob < 0.4:
        return "LOW"
    elif defect_prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"
def log_prediction(image_path, output):
    log_entry = {
        "image": os.path.basename(image_path),
        "result": output["result"],
        "confidence": output["confidence"],
        "severity": output["severity"],
        "data_drift_warning": output["data_drift_warning"],
        "timestamp": datetime.now().isoformat()
    }

    log_file = os.path.join(PROJECT_ROOT, "logs", "predictions.json")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)

def predict(image_path, confidence_threshold=0.65):
    """
    Returns:
        result (str): GOOD / DEFECTIVE / UNCERTAIN
        confidence (float)
        severity (str or None)
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    input_mean = np.array(image).mean()
    drift_warning = False

    if train_mean is not None:
        if abs(input_mean - train_mean) > 15:
            drift_warning = True

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(probs, 1)
    confidence = confidence.item()
    defect_prob = probs[0][1].item()

    if confidence < confidence_threshold:
        result = "UNCERTAIN – HUMAN REVIEW REQUIRED"
        severity = None
    elif predicted_class.item() == 0:
        result = "GOOD"
        severity = None
    else:
        result = "DEFECTIVE"
        severity = severity_level(defect_prob)

    output = {
        "result": result,
        "confidence": round(confidence, 3),
        "severity": severity,
        "data_drift_warning": drift_warning
    }
    log_prediction(image_path, output)
    return output

if __name__ == "__main__":
    test_image = os.path.join(PROJECT_ROOT, "sample_test.jpg") 

    if os.path.exists(test_image):
        output = predict(test_image)
        print("\n INFERENCE RESULT")
        for k, v in output.items():
            print(f"{k}: {v}")
    else:
        print(" inference.py loaded successfully (no test image provided)")
