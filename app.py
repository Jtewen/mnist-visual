from flask import Flask, render_template, request, jsonify
from model import MNISTModel
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import tensorflow as tf
import logging
import os
import argparse


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='MNIST Digit Recognition Server')
parser.add_argument('--train', action='store_true', help='Force model retraining')
args = parser.parse_args()

mnist_model = None

def initialize_model():
    global mnist_model
    mnist_model = MNISTModel()
    try:
        if args.train:
            raise FileNotFoundError("Forcing retraining")
        mnist_model.model.load_weights(mnist_model.model_path)
        logging.info("Model loaded successfully.")
    except:
        logging.info("Training model...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train / 255.0
        mnist_model.train(x_train, y_train, epochs=5)
    
    mnist_model.setup_layer_models()

@app.route('/')
def index():
    filters_dir = 'static/filters'
    if not os.path.exists(filters_dir):
        os.makedirs(filters_dir)
    
    filter_images = []
    if os.path.exists(filters_dir):
        for layer in ['conv1', 'conv2']:
            layer_filters = [f for f in os.listdir(filters_dir) if layer in f]
            layer_filters.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            filter_images.extend(layer_filters)
    
    return render_template('index.html', filter_images=filter_images)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']
        
        img = Image.open(io.BytesIO(base64.b64decode(img_data.split(',')[1])))
        img = img.convert('L').resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = mnist_model.predict(img_array)
        feature_maps = mnist_model.get_feature_maps(img_array)
        
        return jsonify({
            'confidence': {str(i): float(pred) for i, pred in enumerate(predictions[0])},
            'feature_maps': feature_maps
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, use_reloader=False)
