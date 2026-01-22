import os
import cv2
import numpy as np
import tensorflow as tf

# ==============================================================
# CONFIGURATION
# ==============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "my_model.h5")
TEST_DIR = os.path.join(PROJECT_ROOT, "data_set", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_outputs")
FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "final_heatmap_output")  # new folder

IMG_SIZE = (224, 224)
TARGET_LAYER = "Conv_1"

# Class mapping: 0 = DEFECTIVE, 1 = GOOD
CLASS_NAMES = {0: "DEFECTIVE", 1: "GOOD"}

# Create output directories if they don't exist
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# ==============================================================
# LOAD MODEL
# ==============================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(TARGET_LAYER).output, model.output]
)

# ==============================================================
# GRAD-CAM FUNCTION
# ==============================================================
def generate_gradcam(img_input, original_img):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    # ReLU & normalization
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-10)

    # Resize to original image
    cam_resized = cv2.resize(cam.numpy(), (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Highlight top activation areas (localized defect)
    threshold = 0.5  # top 50% activations
    mask = cam_resized >= threshold
    highlight = original_img.copy()
    highlight[mask] = cv2.addWeighted(original_img, 0.3, heatmap, 0.7, 0)[mask]

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    final_output = cv2.addWeighted(overlay, 0.7, highlight, 0.3, 0)

    return final_output

# ==============================================================
# PROCESS DEFECTIVE FOLDER ONLY
# ==============================================================
DEFECTIVE_DIR = os.path.join(TEST_DIR, "defective")
if not os.path.exists(DEFECTIVE_DIR):
    print(f"No defective folder found at {DEFECTIVE_DIR}")
else:
    images = [f for f in os.listdir(DEFECTIVE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    for img_name in images:
        img_path = os.path.join(DEFECTIVE_DIR, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_input = np.expand_dims(img_resized / 255.0, axis=0)

        # Generate Grad-CAM overlay
        overlay = generate_gradcam(img_input, img_bgr)

        # Add text annotation with model prediction
        predictions = model.predict(img_input, verbose=0)
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_idx])
        pred_label = CLASS_NAMES[pred_idx]
        text = f"PRED: {pred_label} ({confidence:.2f})"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save overlay in final_heatmap_output folder
        save_name = f"gradcam_defective_{img_name}"
        cv2.imwrite(os.path.join(FINAL_OUTPUT_DIR, save_name), overlay)

    print(f"\n✔ All defective heatmaps saved in: {FINAL_OUTPUT_DIR}")
