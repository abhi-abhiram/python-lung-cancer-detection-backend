import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io


tf.config.set_visible_devices([], "GPU")

# Load the Keras model
model_path = "./model.keras"
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


def preprocess_image(image_path, target_size=(256, 256)):
    try:
        image = Image.open(image_path)
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = image_array.reshape(1, target_size[0], target_size[1], 1)
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# Replace with the path to your test image
test_image_path = r"./test-images/Bengin case (1).jpg"
processed_image = preprocess_image(test_image_path)

if processed_image is not None:
    try:
        result = model.predict(processed_image)
        print("Prediction result:", result)
    except Exception as e:
        print(f"Error during prediction: {e}")
