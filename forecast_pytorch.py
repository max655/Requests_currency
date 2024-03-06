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

cursor.execute("SELECT * FROM exchange_rates WHERE cc = 'USD' and exchange_date BETWEEN 20240101 AND 20240307 "
               "ORDER BY exchange_date ASC")
data = cursor.fetchall()

df = pd.DataFrame(data, columns=['r030', 'txt', 'rate', 'cc', 'exchange_date'])

df['exchange_date'] = pd.to_datetime(df['exchange_date'], format="%Y%m%d")

df.set_index('exchange_date', inplace=True)

start_index = int(0.8 * len(df))

df['rate'].plot(figsize=(12, 6))
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


# Create sequences
seq_length = 10
sequences, labels = create_sequences(data, seq_length)

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
    for seq, lbls in zip(sequences, labels):
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

last_sequences = sequences[-50:]

last_data_values = df['rate'].tail(seq_length).values

last_data = torch.FloatTensor(last_data_values).view(-1, 1)
future_time_steps = 50
new_predictions = []

# Прогноз для кожного майбутнього часового кроку
with torch.no_grad():
    for _ in range(future_time_steps):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        prediction = model(last_sequences.view(-1, 1)).item()
        new_predictions.append(prediction)

        # Оновлення last_data для використання в наступному кроці
        last_data = torch.cat((last_data[0][1:], torch.tensor([[prediction]])))


# Inverse Transform to get the original scale
# test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))

# test_labels = scaler.inverse_transform(test_labels.numpy().reshape(-1, 1))
new_pr = scaler.inverse_transform(np.array(new_predictions).reshape(-1, 1))
print(new_pr)

# Plotting
plt.plot(new_pr, label='Predicted')
# plt.plot(test_labels, label='True')
plt.legend()
plt.show()
