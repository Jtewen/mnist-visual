from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import numpy as np
import json
import tensorflow as tf
from model_vae import VAE

app = Flask(__name__)

# Initialize VAE by loading the saved model
vae = tf.keras.models.load_model('vae_model', custom_objects={'VAE': VAE})

@app.route('/')
def index():
    return render_template('vae.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    z1 = float(data.get('z1', 0))
    z2 = float(data.get('z2', 0))
    z = [z1, z2]
    generated_img = vae.generate_digit(z)

    # Ensure the image is 2D
    if generated_img.ndim == 3 and generated_img.shape[-1] == 1:
        generated_img = np.squeeze(generated_img, axis=-1)
    elif generated_img.ndim == 3 and generated_img.shape[-1] == 3:
        generated_img = np.dot(generated_img[..., :3], [0.2989, 0.5870, 0.1140])

    # Convert image to base64
    img_pil = Image.fromarray(generated_img, mode='L')
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({'image': f'data:image/png;base64,{img_str}'})

@app.route('/digit_averages')
def digit_averages():
    try:
        with open('digit_averages.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)