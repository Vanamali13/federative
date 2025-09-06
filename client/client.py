import flwr as fl
import tensorflow as tf
import pandas as pd
import argparse
import os

from model import create_model

# Parse command-line arguments to get the data file path
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--data", type=str, required=True, help="Path to the local CSV data file")
args = parser.parse_args()

# Function to load and preprocess the data from the CSV file
def load_data(file_path):
    """Loads a CSV file and prepares it for training."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Assuming the last column is the label and the rest are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert to TensorFlow datasets
    train_data = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y.astype('int32')))
    test_data = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y.astype('int32')))
    
    return train_data, test_data

# Flower client that will handle training and evaluation
class CSVClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        """Returns the local model's parameters."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Trains the model with the received parameters."""
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluates the model with the received parameters."""
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": acc}

# Main script
if __name__ == "__main__":
    # Load the partitioned data for this specific client
    train_ds, test_ds = load_data(args.data)
    
    # Extract data for training and evaluation
    x_train, y_train = next(iter(train_ds.batch(len(train_ds))))
    x_test, y_test = next(iter(test_ds.batch(len(test_ds))))
    
    # Create the model architecture
    model = create_model(input_shape=(x_train.shape[1],))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Start the client with the specified server address
    print("Client connecting to server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=CSVClient(model, x_train, y_train, x_test, y_test).to_client()
    )