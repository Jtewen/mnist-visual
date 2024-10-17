from flask import Flask, render_template, request, jsonify
from model import MNISTModel
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import tensorflow as tf
import logging
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
        mnist_model.train(x_train, y_train, epochs=10)

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    load_and_train()
    app.run(debug=True)
