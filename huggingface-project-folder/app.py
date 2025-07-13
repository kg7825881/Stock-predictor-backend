from fastapi import FastAPI, Query
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
import joblib

app = FastAPI()

# Load model & scaler
model = tf.keras.models.load_model("lstm_multi.h5")
scaler = joblib.load("scaler_multi.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Price Predictor API"}

@app.get("/predict/")
def predict_stock(ticker: str = Query(...), days: int = Query(30)):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            return {"error": "Could not fetch data for the given ticker."}

        # Add technical indicators
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd()
        df = df.dropna()

        # Scale and prepare input
        scaled_data = scaler.transform(df[['Close', 'RSI', 'MACD']])
        X = []
        window_size = 60
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i])
        X = np.array(X)

        last_window = X[-1].reshape(1, 60, 3)

        predictions = []
        current_window = last_window

        for _ in range(days):
            pred = model.predict(current_window)[0][0]
            predictions.append(pred)
            new_row = np.array([[pred, df['RSI'].iloc[-1], df['MACD'].iloc[-1]]])
            current_window = np.append(current_window[:, 1:, :], new_row.reshape(1, 1, 3), axis=1)

        # Inverse transform predictions
        predicted_prices = scaler.inverse_transform(
            np.hstack([np.array(predictions).reshape(-1, 1), 
                       np.zeros((days, 2))])
        )[:, 0]

        return {"ticker": ticker, "predicted_prices": predicted_prices.tolist()}
    
    except Exception as e:
        return {"error": str(e)}