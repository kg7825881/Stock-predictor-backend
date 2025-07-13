from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fetch_data_with_indicators import fetch_data_with_indicators
from preprocess_multifeature import preprocess_multifeature_lstm
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

app = FastAPI(title="Stock Price Predictor")

# Allow all CORS (for frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Stock Price Prediction API"}

@app.get("/predict")
def predict_price(ticker: str = Query(..., example="AAPL"), days: int = 30):
    try:
        # Load model & scaler
        model = tf.keras.models.load_model(f"lstm_model_{ticker}.h5")
        scaler = joblib.load(f"scaler_{ticker}.pkl")
    except:
        return {"error": f"Model or scaler not found for {ticker}. Please train it first."}

    try:
        df = fetch_data_with_indicators(ticker, "2015-01-01")
        data = df[['Close', 'MA20', 'MA50', 'RSI']].values
        scaled = scaler.transform(data)
    except:
        return {"error": f"Could not fetch or preprocess data for {ticker}"}

    seq_len = 60
    input_seq = scaled[-seq_len:].reshape(1, seq_len, 4)
    predictions_scaled = []

    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions_scaled.append(pred)

        last_step = input_seq[0][-1].copy()
        last_step[0] = pred
        input_seq = np.append(input_seq[:, 1:, :], [[last_step]], axis=1)

    # Inverse transform
    predictions = scaler.inverse_transform(
        np.hstack([np.array(predictions_scaled).reshape(-1, 1), np.zeros((days, 3))])
    )[:, 0]

    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=days)
    result = [{"date": str(date.date()), "predicted_price": round(float(price), 2)}
              for date, price in zip(future_dates, predictions)]

    return {
        "ticker": ticker,
        "last_price": round(float(df['Close'].iloc[-1]), 2),
        "last_date": str(df.index[-1].date()),
        "predictions": result
    }