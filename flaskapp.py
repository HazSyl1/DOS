from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained Q-learning classifier and scaler
with open('ddos_classifier_q_table.pkl', 'rb') as model_file:
    q_table = pickle.load(model_file)

# Initialize the scaler
scaler = StandardScaler()

# Q-Learning DDoS Classifier
class QLearningDDoSClassifier:
    def __init__(self, q_table, actions=[0, 1]):
        self.q_table = q_table
        self.actions = actions

    def predict(self, state):
        # Predict based on Q-table
        action = np.argmax(self.q_table[state])  # Get the index with the highest Q-value for the state
        return 'Dos' if action == 1 else 'Not Dos'


# Initialize classifier with the loaded Q-table
classifier = QLearningDDoSClassifier(q_table=q_table)

# Helper function to preprocess the input data (from JSON)
def preprocess_input(json_input):
    input_df = pd.DataFrame([json_input])
    
    # Convert IP addresses and Protocol to numerical
    input_df['src'] = input_df['src'].apply(lambda x: int(x.replace(".", "")))
    input_df['dst'] = input_df['dst'].apply(lambda x: int(x.replace(".", "")))
    input_df['Protocol'] = input_df['Protocol'].map({"TCP": 0, "UDP": 1, "ICMP": 2})

    # Assuming 'X' columns are predefined (as in the original dataset)
    input_features = input_df[['dt', 'switch', 'src', 'dst', 'pktcount', 'bytecount', 'dur', 
                               'dur_nsec', 'tot_dur', 'flows', 'packetins', 'pktperflow', 
                               'byteperflow', 'pktrate', 'Pairflow', 'Protocol', 'port_no', 
                               'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']]

    # Scale the input features using the same scaler used during training
    scaled_input = scaler.transform(input_features)
    
    return scaled_input[0]  # Return the processed single input row


@app.route('/')
def index():
    return "Flask App Running"
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON input from the request
        json_input = request.get_json()

        # Preprocess the input
        processed_input = preprocess_input(json_input)

        # Predict the label
        state = np.argmax(processed_input)  # Get the index of the highest value from the processed input
        prediction_label = classifier.predict(state)  # Get the prediction

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
