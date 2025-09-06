import tensorflow as tf
from tensorflow.keras import layers

def create_model(input_shape):
    """Creates a simple dense model with a flexible input shape."""
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax") # Adjust the output layer for your number of classes
    ])
    return model