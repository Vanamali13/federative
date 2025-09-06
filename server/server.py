import flwr as fl
import tensorflow as tf

from model import create_model

# Define the server strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,  # Minimum number of clients to start a round
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Start the Flower server
print("Starting Federated Server...")

# Create the initial global model
model = create_model(input_shape=(13,)) # Example input shape, adjust based on your data

# Save the initial model to a .h5 file for the API server
model.save("global_model.h5")

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

# After training, the aggregated model will be saved to 'global_model.h5'
print("Training finished. Final model saved to global_model.h5")