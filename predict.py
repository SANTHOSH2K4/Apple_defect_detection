import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np


model = keras.models.load_model("apple_quality_classification.h5")

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_labels = ['defective', 'good']
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

image_path = "2.jpg"
predicted_class = classify_image(image_path)
print("Predicted Class:", predicted_class)
