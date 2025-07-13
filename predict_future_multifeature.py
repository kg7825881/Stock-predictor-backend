# predict_future_multifeature.py
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from fetch_data_with_indicators import fetch_data_with_indicators
from datetime import timedelta
import sys

def predict_future_multifeature(ticker='AAPL', start='2015-01-01', future_days=30, sequence_length=60):
    try:
        model = tf.keras.models.load_model('lstm_multi.h5')
        scaler = joblib.load('scaler_multi.pkl')
    except:
        print("Model or scaler not found. Make sure you've trained the model first.")
        sys.exit(1)

    try:
        df = fetch_data_with_indicators(ticker, start)
    except:
        print(f"Could not fetch data for ticker: {ticker}")
        sys.exit(1)

    last_data = df[['Close', 'MA20', 'MA50', 'RSI']].values
    if len(last_data) < sequence_length:
        print(f"Not enough data to form a sequence of {sequence_length} for {ticker}")
        sys.exit(1)

    last_scaled = scaler.transform(last_data)
    input_seq = last_scaled[-sequence_length:].reshape(1, sequence_length, 4)

    predictions_scaled = []
    for _ in range(future_days):
        next_scaled = model.predict(input_seq, verbose=0)[0][0]
        predictions_scaled.append(next_scaled)

        last_step = input_seq[0][-1].copy()
        last_step[0] = next_scaled  # only replace 'Close'
        input_seq = np.append(input_seq[:, 1:, :], [[last_step]], axis=1)

    # Prepare dummy input for inverse_transform
    predictions = scaler.inverse_transform(np.hstack([
        np.array(predictions_scaled).reshape(-1, 1),
        np.zeros((future_days, 3))
    ]))[:, 0]

    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)

    # ➕ Show last actual closing price
    last_close_price = df['Close'].iloc[-1]
    last_close_date = df.index[-1].strftime('%Y-%m-%d')
    currency = "₹" if ".NS" in ticker or ".BO" in ticker else "$"
    print(f"\n📌 Last available price for {ticker.upper()}: {currency}{float(last_close_price):.2f} on {last_close_date}") 

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-60:], df['Close'][-60:], label='Last 60 Days')
    plt.plot(future_dates, predictions, label='Predicted Prices', linestyle='--', marker='o')
    plt.title(f'{ticker.upper()} Future Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return and print
    result_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    print("\n🔮 Predicted Prices:")
    print(result_df.head(10))
    return result_df

# Run from user input
if __name__ == "__main__":
    print("📈 Stock Price Predictor (Multi-Feature LSTM)")
    user_ticker = input("Enter stock ticker (e.g. AAPL for US, RELIANCE.NS for India): ").strip().upper()
    user_days = input("Enter how many future days to predict (default 30): ").strip()

    future_days = int(user_days) if user_days.isdigit() else 30
    predict_future_multifeature(ticker=user_ticker, future_days=future_days)