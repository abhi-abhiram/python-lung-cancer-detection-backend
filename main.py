import os
import json
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

# Load the Keras model
model_path = "./model.keras"
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load class labels
classes_path = "./classes.json"
try:
    with open(classes_path, "r") as f:
        class_labels = json.load(f)
    print("Class labels loaded successfully:", class_labels)
except Exception as e:
    print(f"Error loading class labels: {e}")
    class_labels = []


def preprocess_image(image, target_size=(256, 256)):
    """Preprocess the image to match training preprocessing steps."""
    try:
        # Convert image to grayscale
        if image.mode != "L":
            image = image.convert("L")
        # Resize the image to the target size
        image = image.resize(target_size)
        # Convert the image to a NumPy array
        image_array = np.array(image, dtype=np.float32)
        # Normalize the pixel values to [0, 1]
        image_array /= 255.0
        # Reshape to match the model's input shape (1, 256, 256, 1)
        image_array = image_array.reshape(1, target_size[0], target_size[1], 1)
        return image_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(256, 256))

        if processed_image is None:
            return jsonify({"error": "Image preprocessing failed"}), 500

        # Perform inference
        result = model.predict(processed_image)

        # Process the result
        # Assuming the model outputs probabilities for each class
        predicted_class_index = np.argmax(result, axis=1)[0]
        predicted_class = (
            class_labels[predicted_class_index]
            if class_labels
            else predicted_class_index
        )

        confidence = float(np.max(result))

        return jsonify(
            {
                "detections": {
                    "class_index": int(predicted_class_index),
                    "class_name": predicted_class,
                    "confidence": confidence,
                }
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Dummy route for testing
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "This is a test route", "classes": class_labels})


if __name__ == "__main__":
    if model:
        app.run(debug=True)
    else:
        print("Model not loaded. Flask app not started.")
