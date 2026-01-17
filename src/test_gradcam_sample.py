import os
import warnings
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Suppress warnings for a clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# 2. Set exact paths based on your folder structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_models', 'my_model.h5')
# Target the specific 'defective' folder in your screenshot
DATASET_DIR = os.path.join(BASE_DIR, '..', 'processed_data', 'test', 'defective')
OUTPUT_PATH = os.path.join(BASE_DIR, 'gradcam_result.jpg')

def run_jan15_test():
    print("--- Member 3: Jan 15 Task (Final Test) ---")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Get first image from the directory
    if not os.path.exists(DATASET_DIR):
        print(f"Directory not found: {DATASET_DIR}")
        return
        
    image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("No images found in the defective folder.")
        return
    
    img_path = os.path.join(DATASET_DIR, image_files[0])
    print(f"Testing on: {img_path}")

    # 3. Grad-CAM Logic
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_rgb, (224, 224)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("out_relu").output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    # 4. Superimpose and Save
    heatmap_res = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    cv2.imwrite(OUTPUT_PATH, superimposed)
    print(f"Predicted Class: {np.argmax(predictions[0])}")
    print(f"SUCCESS: Jan 15th task complete. Heatmap saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_jan15_test()