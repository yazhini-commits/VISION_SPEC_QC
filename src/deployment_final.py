import cv2
import numpy as np
import time
from collections import deque

# =========================================================
# 12 JAN 2026 – OpenCV Installation & Setup
# =========================================================
CAMERA_INDEX = 0
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# =========================================================
# 15 JAN 2026 – FPS Targets & Metrics
# =========================================================
TARGET_FPS = 30
MIN_ACCEPTABLE_FPS = 20
FPS_AVG_WINDOW = 30

# =========================================================
# 16 JAN 2026 – FPS Benchmarking Class
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
        if not self.fps_values:
            return 0.0
        return sum(self.fps_values) / len(self.fps_values)

# =========================================================
# 14 JAN 2026 – Inference Pipeline (Preprocess → Infer → Postprocess)
# =========================================================
def preprocess(frame):
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

def dummy_inference(input_tensor):
    time.sleep(0.01)  # simulate inference time
    return "Object", 0.95

def postprocess(frame, label, confidence):
    cv2.putText(
        frame,
        f"{label}: {confidence:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return frame

# =========================================================
# 13 JAN 2026 – Real-Time Inference Workflow
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

        # Step 1: Preprocess
        input_tensor = preprocess(frame)

        # Step 2: Inference
        label, confidence = dummy_inference(input_tensor)

        # Step 3: Postprocess
        frame = postprocess(frame, label, confidence)

        # Step 4: FPS Benchmarking
        fps_meter.update()
        avg_fps = fps_meter.average()

        # FPS Status (15 Jan Metrics)
        if avg_fps >= TARGET_FPS:
            status = "PASS"
        elif avg_fps >= MIN_ACCEPTABLE_FPS:
            status = "WARN"
        else:
            status = "FAIL"

        cv2.putText(
            frame,
            f"FPS: {avg_fps:.2f} ({status})",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("Vision QC – 12 to 16 Jan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline execution completed")

# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_realtime_pipeline()
