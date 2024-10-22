from flask import Flask, render_template, request, jsonify
from model import MNISTModel
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import tensorflow as tf
import logging
import os

app = Flask(__name__)
mnist_model = MNISTModel()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load MNIST data and train the model if not already trained
def load_and_train():
    try:
        mnist_model.model.load_weights(mnist_model.model_path)
        print("Model loaded successfully.")
    except:
        print("Training model...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train / 255.0
        mnist_model.train(x_train, y_train, epochs=5)

@app.route('/')
def index():
    # List saved filter images
    filter_images = os.listdir('static/filters')
    # Initialize feature_map_images as empty
    feature_map_images = {
        'conv1': [],
        'conv2': []
    }
    return render_template('index.html', filter_images=filter_images, feature_map_images=feature_map_images)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']
    # Decode the image
    img = Image.open(io.BytesIO(base64.b64decode(img_data.split(',')[1])))
    img = img.convert('L').resize((28, 28))
    # Invert image colors
    img = ImageOps.invert(img)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    predictions = mnist_model.predict(img_array)
    confidence = {str(i): float(pred) for i, pred in enumerate(predictions[0])}
    
    return jsonify(confidence)

@app.route('/feature_maps', methods=['POST'])
def feature_maps():
    data = request.get_json()
    img_data = data['image']
    
    # Decode the image
    img = Image.open(io.BytesIO(base64.b64decode(img_data.split(',')[1])))
    img = img.convert('L').resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Ensure the shape is (1, 28, 28, 1)

    # Get feature maps
    feature_maps = mnist_model.get_feature_maps(img_array)

    # Prepare feature maps for rendering
    feature_map_images = {}
    
    for layer_name, fmap in feature_maps.items():
        feature_map_images[layer_name] = []
        # Normalize and convert feature maps to images
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # Normalize to [0, 1]
        
        if fmap.ndim == 4:  # If it's a 4D array (batch, height, width, channels)
            for channel in range(fmap.shape[-1]):
                fmap_image = fmap[0, :, :, channel]  # Use the specific channel
                fmap_image = (fmap_image * 255).astype(np.uint8)  # Scale to [0, 255]
                
                # Create a PIL image
                pil_image = Image.fromarray(fmap_image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Append to the corresponding layer's feature map images
                feature_map_images[layer_name].append(img_str)

    return jsonify(feature_map_images)

if __name__ == '__main__':
    load_and_train()
    app.run(debug=True)
