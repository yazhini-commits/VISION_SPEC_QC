import os
import warnings
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_models', 'my_model.h5')
DATASET_DIR = os.path.join(BASE_DIR, '..', 'processed_data', 'test', 'defective')
OUTPUT_PATH = os.path.join(BASE_DIR, 'gradcam_result_validated.jpg')

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
target_layer = model.get_layer("out_relu")

# List all images in dataset
image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    raise FileNotFoundError(f"No images found in {DATASET_DIR}")

print("Images found:")
for i, f in enumerate(image_files, 1):
    print(f"{i}. {f}")

# Pick the first image
img_path = os.path.join(DATASET_DIR, image_files[0])
print(f"\nValidating Grad-CAM on: {img_path}")

# Preprocess
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = cv2.resize(img_rgb, (224, 224)) / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Grad-CAM logic
grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_input)
    pred_class = np.argmax(predictions[0])
    print(f"Model Prediction Index: {pred_class}, Confidence: {predictions[0][pred_class]:.4f}")
    loss = predictions[:, pred_class]

grads = tape.gradient(loss, conv_outputs)
weights = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
heatmap = heatmap.numpy()

# Overlay
heatmap_res = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

cv2.imwrite(OUTPUT_PATH, superimposed)
plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"Grad-CAM validation complete. Heatmap saved to {OUTPUT_PATH}")
