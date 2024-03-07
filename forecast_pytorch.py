import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('exchange_rates.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM exchange_rates WHERE cc = 'USD' and exchange_date BETWEEN 20140101 AND 20150101 "
               "ORDER BY exchange_date ASC")
data = cursor.fetchall()
for row in data:
    print(row)


df = pd.DataFrame(data, columns=['r030', 'txt', 'rate', 'cc', 'exchange_date'])

df['exchange_date'] = pd.to_datetime(df['exchange_date'], format="%Y%m%d")

df.set_index('exchange_date', inplace=True)

start_index = int(0.8 * len(df))

df['rate'][start_index:].plot(figsize=(12, 6))
plt.title('Time Series Plot')
plt.show()

# Data Preparation
scaler = MinMaxScaler()
df[['rate']] = scaler.fit_transform(df[['rate']])

# Convert DataFrame to PyTorch tensors
data = torch.FloatTensor(df['rate'].values).view(-1, 1)

# Define a function to create input sequences and corresponding labels
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length:i + seq_length + 1]
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.stack(labels)

# Set the sequence length and create sequences
seq_length = 10  # You can adjust this parameter
sequences, labels = create_sequences(data, seq_length)

# Train-Test Split
train_size = int(len(sequences) * 0.8)
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# Instantiate the model, define the loss function and optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    for seq, labels in zip(train_sequences, train_labels):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
test_predictions = []

for seq in test_sequences:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_predictions.append(model(seq).item())

# Inverse Transform to get the original scale
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
test_labels = scaler.inverse_transform(test_labels.numpy().reshape(-1, 1))

# Plotting
plt.plot(test_predictions, label='Predicted')
plt.plot(test_labels, label='True')
plt.legend()
plt.show()
