import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# ==============================
# PATH CONFIGURATION
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "my_model.h5")
TEST_DIR = os.path.join(PROJECT_ROOT, "data_set", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gradcam_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAST_CONV_LAYER = "Conv_1"  # MobileNetV2

# ==============================
# LOAD MODEL
# ==============================
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

# ==============================
# GRAD-CAM FUNCTION
# ==============================
def get_gradcam(img_path, model, last_conv_layer_name, save_path, class_names=["non_defective", "defective"]):
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    # Predict class
    preds = model.predict(img_array, verbose=0)
    pred_label = 1 if preds[0][0] > 0.5 else 0
    confidence = preds[0][0] if pred_label == 1 else 1 - preds[0][0]
    label_text = f"{class_names[pred_label]}: {confidence:.2f}"

    # Grad-CAM computation
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Put predicted label text on image
    cv2.putText(
        superimposed_img, label_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    # Save output
    cv2.imwrite(save_path, superimposed_img)
    print(f"✅ Grad-CAM saved at {save_path}")

# ==============================
# LOOP OVER TEST FOLDERS
# ==============================
for subdir in ["defective", "non_defective"]:
    folder = os.path.join(TEST_DIR, subdir)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found, skipping: {folder}")
        continue

    class_output_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(class_output_dir, exist_ok=True)

    counter = 1
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            save_name = f"gradcam_{counter}.png"
            save_path = os.path.join(class_output_dir, save_name)
            get_gradcam(img_path, model, LAST_CONV_LAYER, save_path)
            counter += 1
