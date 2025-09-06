from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = tf.keras.models.load_model("global_model.h5")
except Exception as e:
    print(f"Model not found. Run server.py to create it. Error: {e}")
    model = None

class PredictionRequest(BaseModel):
    features: list

class WeightsRequest(BaseModel):
    weights: list
    num_samples: int

@app.get("/get-weights")
def get_weights():
    if not model: return {"error": "Model not loaded"}
    weights = model.get_weights()
    weights_list = [w.tolist() for w in weights]
    return {"weights": weights_list}

@app.post("/send-weights")
def send_weights(request: WeightsRequest):
    if not model: return {"error": "Model not loaded"}
    received_weights = [np.array(w) for w in request.weights]
    model.set_weights(received_weights)
    model.save("global_model.h5")
    return {"message": "Weights received and model updated"}
    
# --- THIS IS THE MISSING PART ---
# The /predict endpoint that your HTML page is looking for
@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        return {"error": "Model not loaded on server."}
    
    features = np.array([request.features], dtype=np.float32)
    prediction_probabilities = model.predict(features)
    predicted_class = int(np.argmax(prediction_probabilities, axis=1)[0])
    result_text = "Heart Disease Detected" if predicted_class == 1 else "No Heart Disease"
    
    return {"predicted_class": predicted_class, "result_text": result_text}

@app.get("/")
def read_root():
    return {"message": "Federated Learning API Server is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

