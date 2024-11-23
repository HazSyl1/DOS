from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained Q-learning model (Q-table) and scaler
with open("ddos_classifier_q_table.pkl", "rb") as model_file:
    q_table = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# FastAPI app instance
app = FastAPI()

# Define the input model using Pydantic
class DosInput(BaseModel):
    dt: int
    switch: int
    src: str
    dst: str
    pktcount: int
    bytecount: int
    dur: int
    dur_nsec: int
    tot_dur: float
    flows: int
    packetins: int
    pktperflow: int
    byteperflow: int
    pktrate: int
    Pairflow: int
    Protocol: str
    port_no: int
    tx_bytes: int
    rx_bytes: int
    tx_kbps: float
    rx_kbps: float
    tot_kbps: float

# Function to process and scale the input
def preprocess_input(input_data: DosInput):
    # Convert IP addresses and Protocol to numerical values
    src_int = int(input_data.src.replace(".", ""))
    dst_int = int(input_data.dst.replace(".", ""))
    protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
    protocol_int = protocol_map.get(input_data.Protocol.upper(), -1)

    # Create a numpy array of the input features
    feature_array = np.array([
        input_data.dt, input_data.switch, src_int, dst_int, input_data.pktcount,
        input_data.bytecount, input_data.dur, input_data.dur_nsec, input_data.tot_dur,
        input_data.flows, input_data.packetins, input_data.pktperflow, input_data.byteperflow,
        input_data.pktrate, input_data.Pairflow, protocol_int, input_data.port_no,
        input_data.tx_bytes, input_data.rx_bytes, input_data.tx_kbps, input_data.rx_kbps,
        input_data.tot_kbps
    ]).reshape(1, -1)

    # Scale the input using the pre-loaded scaler
    scaled_input = scaler.transform(feature_array)
    return scaled_input

# Prediction function using the Q-learning model
def predict(input_data: DosInput):
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Predict using Q-table (use the index of the input)
    state_index = 0  # Since we're predicting for a single row, the index is 0
    action = np.argmax(q_table[state_index])

    return "Dos" if action == 1 else "Not Dos"

# Prediction endpoint
@app.post("/predict/")
async def make_prediction(input_data: DosInput):
    try:
        # Make prediction
        prediction = predict(input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# You can add additional endpoints if needed
@app.get("/")
async def read_root():
    return {"message": "Welcome to the DDoS prediction API!"}
