import os
import mlflow
import numpy as np
import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load Model and Scaler
model_path = os.path.join("/app", "Self_healing_network_model_falcon.pkl")
scaler_path = os.path.join("/app", "Network_Scaler.pkl")

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    with open(scaler_path, "rb") as file2:
        scaler = pickle.load(file2)
except FileNotFoundError:
    raise RuntimeError(f"Model files not found at {model_path} or {scaler_path}")


# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow server URL
mlflow.set_experiment("WiFi_Network_Monitoring")  # Your experiment name

class InputData(BaseModel):
    latency: float
    signal_strength: float
    download_speed: float
    upload_speed: float

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.latency, data.signal_strength, data.download_speed, data.upload_speed]])
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)

    # Log metrics in MLflow
    with mlflow.start_run():
        mlflow.log_param("latency", data.latency)
        mlflow.log_param("signal_strength", data.signal_strength)
        mlflow.log_param("download_speed", data.download_speed)
        mlflow.log_param("upload_speed", data.upload_speed)
        mlflow.log_metric("prediction", float(prediction[0]))

    return {"prediction": float(prediction[0])}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
