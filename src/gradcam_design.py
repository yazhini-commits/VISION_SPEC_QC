import tensorflow as tf
import os

# 1. FIX THE PATH (The Design must point to the 'saved_models' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This is the 'src' folder
# Path: Go up one level from 'src', then into 'saved_models'
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_models', 'my_model.h5') 

def complete_jan13_design():
    """
    Member 3 - Task 1: Initial Design of Grad-CAM approach.
    Goal: Define the architecture that bridges the model and the heatmap logic.
    """
    print(f"Searching for model at: {os.path.abspath(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find my_model.h5 in the 'saved_models' folder.")
        print("Check if the folder name is 'saved_models' or 'saved_model'.")
        return

    # 2. LOAD THE TRAINED MODEL
    print("Loading model for design validation...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. DESIGN THE GRAD-CAM EXTRACTOR (Requirement for Jan 13)
    # We identify the last conv layer 'out_relu' to maintain spatial mapping.
    target_layer_name = "out_relu"
    
    # Create the dual-output design: [Feature Maps, Final Predictions]
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    print("\n" + "="*40)
    print("JAN 13 TASK COMPLETE: INITIAL DESIGN")
    print("="*40)
    print(f"Target Layer for Explainability: {target_layer_name}")
    print("Strategy: Extracting gradients from last Conv layer.")
    print("="*40 + "\n")
    
    # Print summary to verify the extractor design is correct
    grad_model.summary()

if __name__ == "__main__":
    complete_jan13_design()