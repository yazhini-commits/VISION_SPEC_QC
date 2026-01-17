import os
import cv2
import numpy as np
import tensorflow as tf

# ==============================================================
# 1. CONFIGURATION (JAN 17 - FINAL INTEGRATION)
# ==============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths - Using data_set/test for human-interpretable results
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "my_model.h5")
TEST_DIR = os.path.join(PROJECT_ROOT, "data_set", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_outputs")

IMG_SIZE = (224, 224)
TARGET_LAYER = "Conv_1" 

# Mapping the numeric codes to human labels
CLASS_NAMES = {0: "DEFECTIVE", 1: "GOOD"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# 2. LOAD MODEL & GRAD-CAM WRAPPER
# ==============================================================
print("Loading model for final audit...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(TARGET_LAYER).output, model.output]
)

# ==============================================================
# 3. GRAD-CAM FUNCTION (ALIGNED FOR VISUAL VALIDATION)
# ==============================================================
def generate_gradcam(img_input, original_img):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    # ReLU and Normalization
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-10)

    # Scale to original image size for better interpretability
    cam_resized = cv2.resize(cam.numpy(), (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return overlay, int(pred_idx), float(predictions[0][pred_idx])

# ==============================================================
# 4. BATCH AUDIT - PROCESSING TEST FOLDERS
# ==============================================================
print(f" Validating on original data: {TEST_DIR}")

for category in ["good", "defective"]:
    category_path = os.path.join(TEST_DIR, category)
    if not os.path.exists(category_path): continue

    images = [f for f in os.listdir(category_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in images:
        img_path = os.path.join(category_path, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        # Resize for model, keep BGR for overlay
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_input = np.expand_dims(img_resized / 255.0, axis=0)

        # Generate Explanation
        overlay, pred_idx, confidence = generate_gradcam(img_input, img_bgr)

        # GET HUMAN LABELS
        pred_label = CLASS_NAMES.get(pred_idx, "UNKNOWN")
        actual_label = category.upper()

        # COLOR LOGIC: Green if Correct, Red if Wrong
        is_correct = pred_label.lower() == category.lower()
        color = (0, 255, 0) if is_correct else (0, 0, 255)

        # TEXT STAMP
        text = f"ACTUAL: {actual_label} | PRED: {pred_label} ({confidence:.2f})"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # SAVE TO OUTPUT
        save_name = f"final_{category}_{img_name}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), overlay)

print(f"\n Audit Complete. Results in: {OUTPUT_DIR}")