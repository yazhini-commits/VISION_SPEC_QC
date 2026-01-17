import cv2
import numpy as np
import time
import os
import tensorflow as tf
from collections import deque

# =========================================================
# CONFIGURATION
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "my_model.h5")

# =========================================================
# OPENCV SETUP
# =========================================================
CAMERA_INDEX = 0
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# =========================================================
# FPS TARGETS
# =========================================================
TARGET_FPS = 30
MIN_ACCEPTABLE_FPS = 20
FPS_AVG_WINDOW = 30

# =========================================================
# MODEL LOADING
# =========================================================
print("Looking for model at:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(frame):
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

# =========================================================
# INFERENCE
# =========================================================
def infer(frame):
    input_tensor = preprocess(frame)

    pred = model.predict(input_tensor, verbose=0)[0][0]

    label = "DEFECT" if pred > 0.5 else "PASS"
    confidence = float(pred if pred > 0.5 else 1.0 - pred)

    return label, confidence

# =========================================================
# FPS BENCHMARKING
# =========================================================
class FPSBenchmark:
    def __init__(self, window=30):
        self.prev_time = None
        self.fps_values = deque(maxlen=window)

    def update(self):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0

        fps = 1.0 / (now - self.prev_time)
        self.prev_time = now
        self.fps_values.append(fps)
        return fps

    def average(self):
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0.0

# =========================================================
# POSTPROCESS
# =========================================================
def postprocess(frame, label, confidence):
    color = (0, 255, 0) if label == "PASS" else (0, 0, 255)

    cv2.putText(
        frame,
        f"{label}: {confidence:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )
    return frame

# =========================================================
# REAL-TIME PIPELINE
# =========================================================
def run_realtime_pipeline():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("ERROR: Camera not accessible")

    fps_meter = FPSBenchmark(FPS_AVG_WINDOW)

    print("Running pipeline (Press 'q' to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        label, confidence = infer(frame)

        # Draw results
        frame = postprocess(frame, label, confidence)

        # FPS
        fps_meter.update()
        avg_fps = fps_meter.average()

        if avg_fps >= TARGET_FPS:
            status = "PASS"
            status_color = (0, 255, 0)
        elif avg_fps >= MIN_ACCEPTABLE_FPS:
            status = "WARN"
            status_color = (0, 255, 255)
        else:
            status = "FAIL"
            status_color = (0, 0, 255)

        cv2.putText(
            frame,
            f"FPS: {avg_fps:.2f} ({status})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        cv2.imshow("Vision QC  Production Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("execution completed")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_realtime_pipeline()
