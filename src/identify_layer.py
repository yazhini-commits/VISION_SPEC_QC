import os
import logging
import warnings

# -------------------------
# FULL SUPPRESSION OF TF MESSAGES
# -------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress INFO/WARNING/ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # disable oneDNN floating point warnings
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'    # suppress absl warnings
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore")

import tensorflow as tf

# -------------------------
# Set paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_models', 'my_model.h5')

def run_jan14_task():
    print("--- Member 3: Jan 14 Task Execution ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Identify target layer
    target_layer_name = "out_relu"
    
    try:
        target_layer = model.get_layer(target_layer_name)
        print(f"Successfully Isolated Layer: {target_layer.name}")
        print(f"Layer Shape: {target_layer.output.shape}")
        
        if len(target_layer.output.shape) == 4:
            print("Status: VALID - Layer maintains spatial mapping for heatmap generation.")
        else:
            print("Status: INVALID - Selected layer does not have spatial dimensions.")
            
    except ValueError:
        print(f"Error: Layer '{target_layer_name}' not found in the model architecture.")

if __name__ == "__main__":
    run_jan14_task()
