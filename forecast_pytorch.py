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

cursor.execute("SELECT * FROM exchange_rates WHERE cc = 'USD' and exchange_date BETWEEN 20180505 AND 20190505 "
               "ORDER BY exchange_date ASC")
data = cursor.fetchall()

df = pd.DataFrame(data, columns=['r030', 'txt', 'rate', 'cc', 'exchange_date'])

df['exchange_date'] = pd.to_datetime(df['exchange_date'], format="%Y%m%d")

df.set_index('exchange_date', inplace=True)

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
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return torch.stack(sequences), torch.stack(labels)


# Set the sequence length and create sequences
seq_length = 10
sequences, labels = create_sequences(data, seq_length)

# Train-Test Split
train_size = int(len(sequences) * 0.7)
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
epochs = 500
for epoch in range(epochs):
    losses = []
    for seq, labels in zip(train_sequences, train_labels):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        losses.append(single_loss.item())

    if epoch % 10 == 0:
        train_rmse = np.sqrt(np.mean(losses))
        print(f'Epoch {epoch}/{epochs}, Train RMSE: {train_rmse:.6f}')

# Evaluate the model
model.eval()
test_predictions = []
test_losses = []

for seq in test_sequences:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        output = model(seq)

        test_loss = criterion(output, labels)
        test_losses.append(test_loss.item())
        test_predictions.append(output.item())

test_rmse = np.sqrt(np.mean(test_losses))
print(f'Test RMSE: {test_rmse:.6f}')


def predict_future(model, initial_sequence, future_steps):
    model.eval()
    prediction_sequence = initial_sequence.clone().detach()
    predictions = []
    for _ in range(future_steps):
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            prediction = model(prediction_sequence)
            predictions.append(prediction.item())

            next_input = torch.cat((prediction_sequence[1:], prediction.view(1, 1)), dim=0)
            prediction_sequence = next_input
    return predictions


initial_sequence = train_sequences[-1].clone().detach()

future_steps = 100
future_predictions = predict_future(model, initial_sequence, future_steps)

# Inverse Transform to get the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
test_labels = scaler.inverse_transform(test_labels.numpy().reshape(-1, 1))
full_data = scaler.inverse_transform(data.numpy())

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(full_data, label='Original Data')
plt.plot(np.arange(len(train_sequences), len(train_sequences) + len(test_labels)),
         test_predictions,
         label='Test Predictions')
plt.plot(np.arange(len(train_sequences), len(train_sequences) + future_steps),
         future_predictions,
         label='Future Predictions', linestyle='--', color='m')
plt.axvline(x=len(train_sequences), color='r', linestyle='--', label='Train-Test Split')
plt.legend()
plt.show()
