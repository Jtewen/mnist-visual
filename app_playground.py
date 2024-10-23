from flask import Flask, render_template, request, jsonify
from model_playground import Playground
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

# Parse command line arguments
parser = argparse.ArgumentParser(description='MNIST Model Playground')
parser.add_argument('--train', action='store_true', help='Force model retraining')
args = parser.parse_args()

# Initialize model
model = None

def initialize_model():
    global model
    model = Playground()
    try:
        if args.train:
            raise FileNotFoundError("Forcing retraining")
        model.model.load_weights(model.model_path)
        logging.info("Model loaded successfully.")
    except:
        logging.info("Training model...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train / 255.0
        model.train(x_train, y_train, epochs=5)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = data['image']
        # Decode the image
        img = Image.open(io.BytesIO(base64.b64decode(img_data.split(',')[1])))
        img = img.convert('L').resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        predictions = model.predict(img_array)
        
        return jsonify({
            'confidence': {str(i): float(pred) for i, pred in enumerate(predictions[0])}
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, use_reloader=False)
