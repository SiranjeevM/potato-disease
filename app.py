import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Function to load the custom KerasLayer
def load_model_with_custom_objects(model_path):
    custom_objects = {"KerasLayer": hub.KerasLayer}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

app = Flask(__name__)

# Ensure the "uploads" directory exists
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

model = load_model_with_custom_objects("model1.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file provided"
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Rescale pixel values
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Input image shape: {img_array.shape}")  # Debugging
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        
        class_names = ["Early blight", "Late blight", "healthy"]
        prediction = class_names[predicted_class]
        
        print(f"Predictions: {predictions}")  # Debugging
        
        # Delete the uploaded file
        os.remove(file_path)
        
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

