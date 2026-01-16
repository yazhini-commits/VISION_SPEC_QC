from inference import predict
import os

# Robust path handling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# CHANGE THIS to any real image you want
image_path = os.path.join(
    PROJECT_ROOT,
    "processed_data",
    "test",
    "good",
    os.listdir(os.path.join(PROJECT_ROOT, "processed_data", "test", "good"))[0]
)

result = predict(image_path)

print("\nDEMO INFERENCE OUTPUT")
for key, value in result.items():
    print(f"{key}: {value}")
