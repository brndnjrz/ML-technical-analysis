
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pandas_ta as ta
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


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

def predict_next_day_close(data, fundamentals, selected_indicators, model_type="RandomForest"):
    """Predicts the next day's closing price using a RandomForestRegressor."""
    try:
        for key, value in fundamentals.items():
            data[key] = value
        # Calculate selected indicators
        if "RSI" in selected_indicators:
            data["RSI"] = ta.rsi(data["Close"], length=14)
        if "MACD" in selected_indicators:
            macd = ta.macd(data["Close"])
            data["MACD"] = macd["MACD_12_26_9"]
            data["MACDh"] = macd["MACDh_12_26_9"]
            data["MACDs"] = macd["MACDs_12_26_9"]
        # Add more indicators as needed
        feature_cols = [col for col in data.columns if col not in ['Close', 'Target']]
        data['Target'] = data['Close'].shift(-1)
        train_data = data.dropna(subset=['Target'])
        X = train_data[feature_cols]
        y = train_data['Target']
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Model selection logic
        model = None
        if model_type == "RandomForest":
            model = RandomForestRegressor()
        elif model_type == "XGBoost":
            if XGBRegressor is not None:
                model = XGBRegressor()
            else:
                st.error("XGBoost is not installed. Please install xgboost to use this model.")
                return None
        elif model_type == "LightGBM":
            if LGBMRegressor is not None:
                model = LGBMRegressor()
            else:
                st.error("LightGBM is not installed. Please install lightgbm to use this model.")
                return None
        elif model_type == "CatBoost":
            if CatBoostRegressor is not None:
                model = CatBoostRegressor(verbose=0)
            else:
                st.error("CatBoost is not installed. Please install catboost to use this model.")
                return None
        else:
            st.error(f"Unknown model_type: {model_type}")
            return None

        model.fit(X_train, y_train)
        last_row = X.iloc[[-1]]
        predicted_price = model.predict(last_row)[0]
        return predicted_price
    except Exception as e:
        st.error(f"❌ Error in predict_next_day_close: {str(e)}")
        return None
