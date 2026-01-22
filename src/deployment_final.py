import cv2
import numpy as np
import os
import time
import tensorflow as tf
from threading import Thread, Lock
from collections import deque

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "my_model.h5")

IMG_SIZE = (224, 224)

TARGET_FPS = 30
MIN_FPS = 20
FPS_WINDOW = 30
MAX_FRAMES = 300

# HD OUTPUT SETTINGS
DISPLAY_SIZE = (1920, 1080)

# =========================================================
# DATASET MODE CONFIG
# =========================================================
DATASET_DIR = os.path.join(BASE_DIR, "processed_data", "test")

# =========================================================
# PREDICTION TUNING
# =========================================================
GOOD_THRESHOLD = 0.7

# =========================================================
# MODEL LOADING
# =========================================================
print("Looking for model at:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# TensorFlow warm-up
dummy = np.zeros((1, IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
model(dummy, training=False)

# =========================================================
# PREPROCESSING
# =========================================================
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

# =========================================================
# INFERENCE
# =========================================================
@tf.function
def model_infer(input_tensor):
    return model(input_tensor, training=False)

def infer(frame):
    input_tensor = preprocess(frame)
    prob_good = float(model_infer(input_tensor)[0][0])

    if prob_good >= GOOD_THRESHOLD:
        return "PASS", prob_good
    else:
        return "DEFECT", 1.0 - prob_good

# =========================================================
# FPS BENCHMARKING
# =========================================================
class FPSMeter:
    def __init__(self, window=30):
        self.prev = None
        self.values = deque(maxlen=window)

    def update(self):
        now = time.perf_counter()
        if self.prev is None:
            self.prev = now
            return 0.0

        fps = 1.0 / (now - self.prev)
        self.prev = now
        self.values.append(fps)
        return fps

    def avg(self):
        return sum(self.values) / len(self.values) if self.values else 0.0

# =========================================================
# FPS VALIDATION
# =========================================================
def fps_status(fps):
    if fps >= TARGET_FPS:
        return "PASS"
    elif fps >= MIN_FPS:
        return "WARN"
    else:
        return "FAIL"

# =========================================================
# HUD PANEL
# =========================================================
def draw_hud(frame, lines, x=30, y=30, width=700, line_height=55):
    overlay = frame.copy()
    height = line_height * len(lines) + 60

    cv2.rectangle(overlay, (x, y), (x + width, y + height), (10, 10, 10), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 3)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        "VISION QC LIVE DASHBOARD",
        (x + 15, y + 45),
        font,
        1.8,
        (0, 255, 255),
        3,
        cv2.LINE_8
    )

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 15, y + 90 + i * line_height),
            font,
            1.6,
            (255, 255, 255),
            3,
            cv2.LINE_8
        )

# =========================================================
# DATASET STREAM
# =========================================================
class DatasetStream:
    def __init__(self, root_folder):
        self.lock = Lock()
        self.stopped = False
        self.samples = []

        for class_name in ["good", "defective"]:
            class_path = os.path.join(root_folder, class_name)
            if not os.path.exists(class_path):
                raise ValueError(f"Missing class folder: {class_name}")

            for file in os.listdir(class_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append({
                        "path": os.path.join(class_path, file),
                        "gt": class_name.upper()
                    })

        if not self.samples:
            raise ValueError("No images found in dataset")

        print(f"Loaded {len(self.samples)} images")

        self.index = 0
        self.frame = cv2.imread(self.samples[self.index]["path"])
        self.current_name = os.path.basename(self.samples[self.index]["path"])
        self.current_gt = self.samples[self.index]["gt"]

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            time.sleep(0.03)
            with self.lock:
                self.index = (self.index + 1) % len(self.samples)
                sample = self.samples[self.index]
                self.frame = cv2.imread(sample["path"])
                self.current_name = os.path.basename(sample["path"])
                self.current_gt = sample["gt"]

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_meta(self):
        with self.lock:
            return self.current_name, self.current_gt

    def stop(self):
        self.stopped = True
        time.sleep(0.2)

# =========================================================
# LIVE DEPLOYMENT LOOP
# =========================================================
def run_live_demo():
    stream = DatasetStream(DATASET_DIR).start()
    time.sleep(1.0)

    fps_meter = FPSMeter(FPS_WINDOW)
    frame_id = 0

    print("VisionSpec QC – Live Deployment")
    print("Mode: 2-Class Dataset Validation")
    print("Press 'q' to exit")

    while True:
        frame = stream.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC)
        filename, gt = stream.get_meta()

        label, confidence = infer(frame)

        fps_meter.update()
        avg_fps = fps_meter.avg()
        status = fps_status(avg_fps)

        hud_lines = [
            f"Image: {filename}",
            f"GT: {gt}  Pred: {label} ({confidence:.2f})",
            f"FPS: {avg_fps:.2f} ({status})"
        ]

        draw_hud(frame, hud_lines)

        cv2.imshow("VisionSpec QC – Live Demo (HD)", frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q') or frame_id >= MAX_FRAMES:
            break

    stream.stop()
    cv2.destroyAllWindows()
    print("Deployment completed")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_live_demo()
