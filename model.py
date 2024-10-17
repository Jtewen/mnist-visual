import tensorflow as tf
from tensorflow.keras import layers, models

class MNISTModel:
    def __init__(self, model_path='mnist_model.h5'):
        self.model_path = model_path
        self.model = self.build_model()
        try:
            self.model.load_weights(self.model_path)
            print("Model weights loaded.")
        except:
            print("No saved model found. Please train the model.")

    def build_model(self):
        model = models.Sequential([
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, epochs=10):
        self.model.fit(x_train, y_train, epochs=epochs)
        self.model.save_weights(self.model_path)
        print(f"Model weights saved to {self.model_path}.")

    def predict(self, image):
        return self.model.predict(image)
