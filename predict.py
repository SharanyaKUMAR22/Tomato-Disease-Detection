import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = tf.keras.models.load_model("tomato_disease_model.keras")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

print("Model loaded successfully!")

# Example prediction function
def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = preds[0][idx] * 100

    print(f"Prediction: {class_names[idx]}")
    print(f"Confidence: {confidence:.2f}%")
