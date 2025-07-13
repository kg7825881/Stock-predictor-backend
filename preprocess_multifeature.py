# preprocess_multifeature.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_multifeature_lstm(df, sequence_length=60):
    features = df[['Close', 'MA20', 'MA50', 'RSI']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, 0])  # Predict only 'Close'

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler