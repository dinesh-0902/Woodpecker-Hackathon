import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

df = pd.read_csv('./raindata.csv')

rainfall_data = df['ACTUAL(mm)'].values

scaler = MinMaxScaler(feature_range=(0, 1))
rainfall_data_scaled = scaler.fit_transform(rainfall_data.reshape(-1, 1))

def create_sequences(data, seq_length=4):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(rainfall_data_scaled, seq_length=4)

X = X.reshape((X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(4, 1), return_sequences=True))
Dropout(0.2)
model.add(LSTM(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

earlyStopping = EarlyStopping(monitor = 'val_loss',patience = 5, restore_best_weights = True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data = (X_test,y_test), callbacks = [earlyStopping])

model.save('rainfall_predictor.h5')
