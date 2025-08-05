import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def get_fundamental_metrics(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        return {
            "EPS": info.get("trailingEps"),
            "P/E Ratio": info.get("trailingPE"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Profit Margin": info.get("profitMargins"),
        }
    except Exception as e:
        st.warning(f"Could not fetch fundamentals: {e}")
        return {}

def predict_next_day_close(data, fundamentals):
    """Predicts the next day's closing price using a RandomForestRegressor."""
    try:
        for key, value in fundamentals.items():
            data[key] = value
        feature_cols = [col for col in data.columns if col not in ['Close', 'Target']]
        data['Target'] = data['Close'].shift(-1)
        train_data = data.dropna(subset=['Target'])
        X = train_data[feature_cols]
        y = train_data['Target']
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        last_row = X.iloc[[-1]]
        predicted_price = model.predict(last_row)[0]
        return predicted_price
    except Exception as e:
        st.error(f"❌ Error in predict_next_day_close: {str(e)}")
        return None
