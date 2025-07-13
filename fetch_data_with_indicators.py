# fetch_data_with_indicators.py
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def fetch_data_with_indicators(ticker: str, start: str):
    df = yf.download(ticker, start=start, auto_adjust=False)
    df = df[['Close']].dropna()

    close_series = df['Close'].squeeze()
    df['MA20'] = SMAIndicator(close_series, window=20).sma_indicator()
    df['MA50'] = SMAIndicator(close_series, window=50).sma_indicator()
    df['RSI'] = RSIIndicator(close_series, window=14).rsi()

    return df.dropna()