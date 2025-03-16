import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import numpy as np

def get_asset_tickers():
    return {
        "Apple Inc. (AAPL)": "AAPL",
        "Tesla Inc. (TSLA)": "TSLA",
        "USD/UAH (Курс долара до гривні)": "USDUAH=X",
        "Gold (Золото)": "GC=F",
        "Cocoa Futures (Ф'ючерси на какао)": "CC=F",
        "Coffee Futures (Ф'ючерси на каву)": "KC=F",
        "Bitcoin to USD (Біткоїн за долари)": "BTC-USD",
        "Ethereum to USD (Ефір за долари)": "ETH-USD",
        "S&P 500 Index (SP500)": "^GSPC",
        "Real Estate Investment Trust (VNQ)": "VNQ"
    }

def get_stock_data(tickers, retries=3, delay=5):
    for i in range(retries):
        try:
            data = yf.download(list(tickers.values()), period="360d", interval="1d", auto_adjust=True)
            if not data.empty:
                return data["Close"].dropna()
        except Exception as e:
            time.sleep(delay)
    return None

def plot_price_dynamics(df, show):
    st.subheader("Динаміка цін активів")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    date_index = df.index
    midpoint = len(date_inde