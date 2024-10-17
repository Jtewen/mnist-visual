# MNIST Classifier and VAE Visualization Demo

This project was created for a Precision Planting keynote presentation. It is a web-based application that allows users to draw digits and get predictions from a trained MNIST model, as well as visualize the latent space of a VAE trained on the MNIST dataset.

## Prerequisites

- Python 3.x
- TensorFlow
- Flask
- Node.js (for serving static files)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jtewen/mnist-visual.git
   cd mnist-visual
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Train Model / Start the Flask server:
   ```bash
   python model.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Draw a digit in the canvas and click the "Clear" button to reset the canvas.

4. The application will automatically send the drawn image to the model for prediction, and the confidence scores will be displayed.

5. **VAE Visualization**:
   - VAE section allows you to visualize the latent space.

## License

This project is licensed under the MIT License.
