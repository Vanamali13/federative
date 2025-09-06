# server/api_server.py

from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add your new origin to this list
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:5500", # <-- ADD THIS LINE
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... the rest of your code remains the same ...
# Load the trained model
model = tf.keras.models.load_model("global_model.h5")

class WeightsRequest(BaseModel):
    weights: list
    num_samples: int

@app.get("/get-weights")
def get_weights():
    """Returns the weights of the global model."""
    global model
    weights = model.get_weights()
    weights_list = [w.tolist() for w in weights]
    return {"weights": weights_list}

@app.post("/send-weights")
def send_weights(request: WeightsRequest):
    """Receives updated weights from a client."""
    global model
    received_weights = [np.array(w) for w in request.weights]
    model.set_weights(received_weights)
    model.save("global_model.h5")
    return {"message": "Weights received and model updated"}

@app.get("/")
def read_root():
    return {"message": "Federated Learning API Server is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)