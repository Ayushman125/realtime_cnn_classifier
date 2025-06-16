import numpy as np
import tensorflow as tf
import cv2 # Make sure to import opencv-python

def load_and_preprocess_mnist(image_size=(28, 28)):
    """
    Loads the MNIST dataset and preprocesses it.
    - Resizes images (if necessary).
    - Normalizes pixel values to [0, 1].
    - Adds channel dimension (for Conv2D).
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Resize and normalize
    x_train = x_train.reshape(x_train.shape[0], image_size[0], image_size[1], 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], image_size[0], image_size[1], 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)

def preprocess_frame(frame, image_size=(28, 28)):
    """
    Preprocesses a single webcam frame for inference.
    - Converts to grayscale (if needed for MNIST).
    - Resizes to model input size.
    - Normalizes pixel values.
    - Adds batch and channel dimensions.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # For MNIST
    resized_frame = cv2.resize(gray_frame, image_size)
    normalized_frame = resized_frame.astype('float32') / 255.0
    # Add channel and batch dimensions: (1, height, width, 1) for Keras
    return np.expand_dims(normalized_frame, axis=[0, -1])