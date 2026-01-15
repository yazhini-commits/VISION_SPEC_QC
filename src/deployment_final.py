import cv2
import numpy as np
import time
import csv
from collections import deque

# =========================================================
# CONFIGURATION
# =========================================================
CAMERA_INDEX = 0
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# 15 Jan – FPS Targets & Metrics
TARGET_FPS = 30
MIN_ACCEPTABLE_FPS = 20

# 16 Jan – FPS Benchmarking
FPS_WINDOW = 30
FPS_LOG_FILE = "fps_benchmark.csv"

# =========================================================
# DUMMY MODEL
# =========================================================
class DummyModel:
    def infer(self, input_tensor):
        time.sleep(0.01)  # simulate inference
        return {"label": "object", "confidence": 0.95}

model = DummyModel()

# =========================================================
# PREPROCESSING (14 Jan)
# =========================================================
def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# =========================================================
# POSTPROCESSING
# =========================================================
def postprocess(frame, output):
    text = f"{output['label']}: {output['confidence']:.2f}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    return frame

# =========================================================
# FPS COUNTER + BENCHMARKING (16 Jan)
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

    def average_fps(self):
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0.0

    def min_fps(self):
        return min(self.fps_values) if self.fps_values else 0.0

    def max_fps(self):
        return max(self.fps_values) if self.fps_values else 0.0

fps_benchmark = FPSBenchmark(FPS_WINDOW)

# =========================================================
# FPS VALIDATION (15 Jan)
# =========================================================
def validate_fps(fps):
    if fps >= TARGET_FPS:
        return "PASS"
    elif fps >= MIN_ACCEPTABLE_FPS:
        return "WARN"
    else:
        return "FAIL"

# =========================================================
# REAL-TIME INFERENCE LOOP (13 Jan)
# =========================================================
def run_realtime_inference():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    # Initialize CSV
    with open(FPS_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "FPS", "Avg FPS", "Status"])

    frame_id = 0
    print("INFO: Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pipeline
        input_tensor = preprocess(frame)
        output = model.infer(input_tensor)
        frame = postprocess(frame, output)

        # FPS Benchmark
        fps = fps_benchmark.update()
        avg_fps = fps_benchmark.average_fps()
        status = validate_fps(avg_fps)

        # Overlay
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        # Log FPS
        with open(FPS_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([frame_id, f"{fps:.2f}", f"{avg_fps:.2f}", status])

        cv2.imshow("Real-Time Inference", frame)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Benchmark completed")
    print(f"FPS log saved to: {FPS_LOG_FILE}")
    print(f"Average FPS: {fps_benchmark.average_fps():.2f}")
    print(f"Min FPS: {fps_benchmark.min_fps():.2f}")
    print(f"Max FPS: {fps_benchmark.max_fps():.2f}")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_realtime_inference()
