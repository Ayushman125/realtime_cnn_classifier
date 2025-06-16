import tensorflow as tf
import matplotlib.pyplot as plt # Already imported in plots.py, but good to have here for direct execution
from src.data_loader import load_and_preprocess_mnist
from src.model import build_cnn_model
from utils.plots import plot_history

def train_model():
    """
    Loads data, builds, trains, and saves the CNN model.
    """
    # Ensure correct input shape for MNIST (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist(image_size=(28, 28))

    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=y_train.shape[1])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test),
                        batch_size=32)
    print("Training complete.")

    # Save the trained model
    model.save('models/trained_cnn_model.h5')
    print("Model saved to models/trained_cnn_model.h5")

    # Plot training history
    plot_history(history)

if __name__ == "__main__":
    train_model()