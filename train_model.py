import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Q-Learning DDoS Classifier
class QLearningDDoSClassifier:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=100, actions=[0, 1]):
        self.alpha = alpha            # Learning rate
        self.gamma = gamma            # Discount factor
        self.epsilon = epsilon        # Exploration rate
        self.episodes = episodes      # Training episodes
        self.actions = actions        # Actions: 0 = BENIGN, 1 = DDoS
        self.q_table = None           # Initialize Q-table

    def fit(self, X_train, y_train):
        # Initialize Q-table with zeros
        self.q_table = np.zeros((X_train.shape[0], len(self.actions)))
        
        # Training loop
        for episode in range(self.episodes):
            for i, (state, label) in enumerate(zip(X_train, y_train)):
                # Epsilon-greedy action selection
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.choice(self.actions)  # Exploration
                else:
                    action = np.argmax(self.q_table[i])      # Exploitation

                # Get reward for the selected action
                reward = 1 if action == label else -1

                # Q-value update
                next_max = np.max(self.q_table[i])  # Max Q-value for next state
                self.q_table[i, action] += self.alpha * (reward + self.gamma * next_max - self.q_table[i, action])
        print("Training complete. Q-table updated.")

    def predict(self, state):
        # Predict based on Q-table
        action = np.argmax(self.q_table[state])  # Get the index with the highest Q-value for the state
        return 'Dos' if action == 1 else 'Not Dos'


# Load the dataset
df = pd.read_csv("dataset_sdn.csv")

# Data Preprocessing
df.columns = df.columns.str.strip()  # Remove spaces in column names
df['label'] = df['label'].map({0: 0, 1: 1})  # Convert labels to numerical values if necessary

# Handle IP addresses and Protocol column (convert to numerical)
df['src'] = df['src'].apply(lambda x: int(x.replace(".", "")))
df['dst'] = df['dst'].apply(lambda x: int(x.replace(".", "")))
df['Protocol'] = df['Protocol'].map({"TCP": 0, "UDP": 1, "ICMP": 2})

# Remove null values
data_f = df.dropna()

# Check for infinite values and replace them with NaN
data_f.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values after replacing infinities
data_f.dropna(inplace=True)

# Split data into features and target variable
X = data_f.drop('label', axis=1)  # Features only
y = data_f['label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Q-Learning Classifier
classifier = QLearningDDoSClassifier(alpha=0.1, gamma=0.9, epsilon=0.1, episodes=100)
classifier.fit(X_train_scaled, y_train)

# Save the Q-table model
with open('ddos_classifier_q_table.pkl', 'wb') as model_file:
    pickle.dump(classifier.q_table, model_file)

print("Q-table model saved successfully.")

# Prediction on test set and calculating accuracy
y_pred = []
for state in X_test_scaled:
    prediction_label = classifier.predict(state)
    # Convert prediction from 'Dos'/'Not Dos' to numerical labels (1 for 'Dos' and 0 for 'Not Dos')
    y_pred.append(1 if prediction_label == 'Dos' else 0)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print confusion matrix to understand the model's performance
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Preprocess JSON input to match training data format
def preprocess_input(json_input):
    input_df = pd.DataFrame([json_input])
    
    # Convert IP addresses and Protocol to numerical
    input_df['src'] = input_df['src'].apply(lambda x: int(x.replace(".", "")))
    input_df['dst'] = input_df['dst'].apply(lambda x: int(x.replace(".", "")))
    input_df['Protocol'] = input_df['Protocol'].map({"TCP": 0, "UDP": 1, "ICMP": 2})

    # Select and order columns to match training data
    input_features = input_df[X.columns.intersection(input_df.columns)]
    
    # Scale the input data
    scaled_input = scaler.transform(input_features)
    
    return scaled_input[0]  # Return the processed single input row

# Prediction function using JSON input
def predict_from_json(json_input):
    # Preprocess JSON input to match the training data format
    processed_input = preprocess_input(json_input)
    
    # Predict using the Q-learning classifier
    state = np.argmax(processed_input)  # Get the index of the highest value from the processed input
    prediction_label = classifier.predict(state)  # Predict label
    
    return prediction_label

# Example JSON input
json_input = {
    'dt': 11545,
    'switch': 2,
    'src': '10.0.0.10',
    'dst': '10.0.0.8',
    'pktcount': 52661,
    'bytecount': 54872762,
    'dur': 170,
    'dur_nsec': 694000000,
    'tot_dur': 171000000000.0,
    'flows': 4,
    'packetins': 1943,
    'pktperflow': 9320,
    'byteperflow': 9711440,
    'pktrate': 310,
    'Pairflow': 0,
    'Protocol': 'UDP',
    'port_no': 4,
    'tx_bytes': 3665,
    'rx_bytes': 3413,
    'tx_kbps': 0,
    'rx_kbps': 0.0,
    'tot_kbps': 0.0,
    'label': 1
}

# Make a prediction with JSON input
prediction_label = predict_from_json(json_input)
print("Prediction:", prediction_label)
