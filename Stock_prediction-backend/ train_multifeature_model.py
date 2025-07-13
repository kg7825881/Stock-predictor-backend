# train_multifeature_model.py

import numpy as np
import tensorflow as tf
import joblib
from fetch_data_with_indicators import fetch_data_with_indicators
from preprocess_multifeature import preprocess_multifeature_lstm

def train_and_save_model(ticker, start='2015-01-01'):
    try:
        print(f"\nFetching data for {ticker}...")
        df = fetch_data_with_indicators(ticker, start)

        print(f"Preprocessing data for {ticker}...")
        X, y, scaler = preprocess_multifeature_lstm(df)

        print(f"Training LSTM model for {ticker}...")
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        model.save(f"lstm_model_{ticker}.h5")
        joblib.dump(scaler, f"scaler_{ticker}.pkl")

        print(f"Model and scaler saved for {ticker}.\n")
    except Exception as e:
        print(f"Failed for {ticker}: {e}\n")


if __name__ == "__main__":
    print("Multi-Stock LSTM Trainer")

    tickers = [
        "AAPL",         # US stock
        "TSLA",         # US stock
        "RELIANCE.NS",  # Indian stock (NSE)
        "TCS.NS",       # Indian stock (NSE)
        "INFY.NS",      # Indian stock (NSE)
        "AXISBANK.NS"   # Indian stock (NSE)
    ]

    for ticker in tickers:
        train_and_save_model(ticker)