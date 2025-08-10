import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pandas_ta as ta
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create advanced technical and price action features for prediction."""
    try:
        df = data.copy()
        
        # Price action features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Moving averages and trends
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['Trend_Strength'] = (df['Close'] - df['SMA_20'])/df['SMA_20']
        
        # Momentum indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['RSI_MA'] = ta.sma(df['RSI'], length=5)
        df['RSI_Trend'] = df['RSI'] - df['RSI_MA']
        
        # Volume analysis
        df['Volume_MA'] = ta.sma(df['Volume'], length=20)
        df['Volume_MA'] = df['Volume_MA'].ffill().infer_objects(copy=False)  # Updated for pandas future
        df['Volume_Trend'] = (df['Volume']/df['Volume_MA']).fillna(1.0)  # Default to neutral
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV'] = df['OBV'].ffill().infer_objects(copy=False)  # Updated for pandas future
        
        # Volatility indicators
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        bb = ta.bbands(df['Close'])
        # Handle potential NaN values in Bollinger Bands with safe access
        if bb is not None and not bb.empty:
            bb_upper = bb.get('BBU_5_2.0', pd.Series()).ffill().infer_objects(copy=False)
            bb_lower = bb.get('BBL_5_2.0', pd.Series()).ffill().infer_objects(copy=False)
            bb_middle = bb.get('BBM_5_2.0', pd.Series()).ffill().infer_objects(copy=False)
            if not bb_middle.empty and not (bb_middle == 0).all():
                df['BB_Width'] = ((bb_upper - bb_lower) / bb_middle).fillna(0)
            else:
                df['BB_Width'] = 0
        else:
            df['BB_Width'] = 0
        
        # Fill NaN values with appropriate methods
        # For trend indicators, forward fill is appropriate
        trend_cols = ['SMA_20', 'EMA_20', 'Trend_Strength', 'Volume_MA', 'BB_Width']
        df[trend_cols] = df[trend_cols].ffill().infer_objects(copy=False)  # Updated for pandas future
        
        # For momentum indicators, use median
        momentum_cols = ['RSI', 'RSI_MA', 'RSI_Trend', 'ATR']
        df[momentum_cols] = df[momentum_cols].fillna(df[momentum_cols].median())
        
        # For returns and volatility, fill with 0
        df[['Returns', 'Log_Returns', 'Volatility']] = df[['Returns', 'Log_Returns', 'Volatility']].fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in engineer_features: {str(e)}")
        raise

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    random_forest = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    xgboost = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    catboost = {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False
    }

def create_model(model_type: str, custom_params: Dict[str, Any] = None) -> Any:
    """Create and configure a model with optimal parameters."""
    config = ModelConfig()
    
    models = {
        "RandomForest": (RandomForestRegressor, config.random_forest),
        "XGBoost": (XGBRegressor, config.xgboost),
        "CatBoost": (CatBoostRegressor, config.catboost)
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    model_class, default_params = models[model_type]
    params = {**default_params, **(custom_params or {})}
    
    return model_class(**params)

def analyze_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Analyze and visualize feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
        
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_imp

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

def predict_next_day_close(data: pd.DataFrame, fundamentals: dict, selected_indicators: list, model_type="RandomForest") -> Tuple[float, float]:
    """Predicts the next day's closing price using the selected model."""
    try:
        # Input validation
        if data is None or data.empty:
            logger.error("Input data is None or empty")
            return None, None
        
        if 'Close' not in data.columns:
            logger.error("'Close' column not found in data")
            return None, None
        
        # Check if we have enough data points
        if len(data) < 20:
            logger.error(f"Insufficient data points: {len(data)}. Need at least 20 for reliable predictions.")
            return None, None
        
        # Check for valid Close prices
        close_values = data['Close'].dropna()
        if len(close_values) == 0:
            logger.error("No valid Close prices found")
            return None, None
        
        if (close_values <= 0).any():
            logger.error("Invalid Close prices (<=0) found in data")
            return None, None
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Process fundamentals with validation
        fundamental_cols = []
        for key, value in fundamentals.items():
            try:
                # Convert to numeric and assign to the first row
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.isna(numeric_value):
                    logger.info(f"Could not convert {key} to numeric. Setting to 0.")
                    numeric_value = 0
                
                # Assign the value to all rows
                df[key] = numeric_value
                fundamental_cols.append(key)
                logger.info(f"Added fundamental metric {key} with value {numeric_value}")
            except Exception as e:
                logger.warning(f"Error processing fundamental {key}: {str(e)}. Setting to 0.")
                df[key] = 0
                fundamental_cols.append(key)
        
        # Engineer features with error handling
        try:
            df = engineer_features(df)
        except Exception as feature_error:
            logger.warning(f"Feature engineering failed: {feature_error}. Using basic features.")
            # Fallback to basic features if engineering fails
            df['Returns'] = df['Close'].pct_change().fillna(0)
            df['SMA_5'] = df['Close'].rolling(5).mean().fillna(df['Close'])
            df['SMA_10'] = df['Close'].rolling(10).mean().fillna(df['Close'])
            df['Volume_MA'] = df['Volume'].rolling(5).mean().fillna(df['Volume']) if 'Volume' in df.columns else df['Close'] * 0
        
        # Calculate additional selected indicators with error handling
        if "RSI" in selected_indicators and "RSI" not in df.columns:
            rsi_result = ta.rsi(df["Close"], length=14)
            if rsi_result is not None:
                df = df.assign(RSI=rsi_result)
        if "MACD" in selected_indicators:
            macd = ta.macd(df["Close"])
            if macd is not None and not macd.empty:
                df = df.assign(
                    MACD=macd.get("MACD_12_26_9", pd.Series()).ffill().infer_objects(copy=False),
                    MACDh=macd.get("MACDh_12_26_9", pd.Series()).ffill().infer_objects(copy=False),
                    MACDs=macd.get("MACDs_12_26_9", pd.Series()).ffill().infer_objects(copy=False)
                )
        
        # Handle missing values with detailed logging
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                logger.info(f"Found {nan_count} NaN values in column {col}")
                
                if col in fundamental_cols:
                    # For fundamental metrics, forward fill and then backfill
                    df[col] = df[col].ffill().bfill().fillna(0).infer_objects(copy=False)  # Updated for pandas future
                    logger.info(f"Filled fundamental metric {col} using forward/backward fill")
                elif col in ['Returns', 'Log_Returns', 'Volatility']:
                    df[col] = df[col].fillna(0)
                    logger.info(f"Filled NaN values in {col} with 0")
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled NaN values in {col} with median value: {median_val}")
                    
        # Double-check for any remaining NaN values
        remaining_nans = df.isna().sum()
        if remaining_nans.any():
            logger.warning("Remaining NaN values found in columns:")
            for col in remaining_nans[remaining_nans > 0].index:
                logger.warning(f"{col}: {remaining_nans[col]} NaN values")
            
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['Close', 'Target']]
        
        # Ensure we have at least some features
        if len(feature_cols) == 0:
            logger.error("No feature columns available for prediction")
            return None, None
        
        df = df.assign(Target=df['Close'].shift(-1))
        
        # Filter out any completely empty feature columns
        valid_feature_cols = []
        for col in feature_cols:
            if col in df.columns and not df[col].isna().all():
                valid_feature_cols.append(col)
        
        if len(valid_feature_cols) == 0:
            logger.error("No valid feature columns with data")
            return None, None
            
        feature_cols = valid_feature_cols
        logger.info(f"Using {len(feature_cols)} feature columns for prediction")
        
        # Final validation before model training
        nan_columns = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
        if nan_columns:
            nan_info = {col: df[col].isna().sum() for col in nan_columns}
            nan_locations = {col: df.index[df[col].isna()].tolist() for col in nan_columns}
            logger.error(f"NaN values found in columns: {nan_info}")
            logger.error(f"NaN locations (indices): {nan_locations}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Data still contains NaN values in columns: {nan_columns}")
        
        # Prepare training data with validation
        train_data = df.dropna(subset=['Target'])
        if len(train_data) < 5:
            logger.error(f"Insufficient training data after dropping NaN: {len(train_data)} rows")
            return None, None
        
        X = train_data[feature_cols]
        y = train_data['Target']
        
        # Verify we have enough data for training and testing
        if len(X) < 20:
            logger.error(f"Insufficient data for prediction: {len(X)} samples. Need at least 20.")
            return None, None
        
        # Use a smaller split for small datasets
        if len(X) < 20:
            split_idx = len(X) - 2  # Keep last 2 for testing
        else:
            split_idx = int(len(X) * 0.8)
            
        if split_idx <= 0:
            logger.error("Cannot split data - insufficient samples")
            return None, None
            
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Ensure we have training data
        if len(X_train) == 0:
            logger.error("No training data available after split")
            return None, None
        
        # Create and train model with error handling
        try:
            model = create_model(model_type)
            model.fit(X_train, y_train)
        except Exception as model_error:
            logger.error(f"Model training failed: {model_error}")
            return None, None
        
        # Make prediction
        last_row = X.iloc[[-1]]
        predicted_price = model.predict(last_row)[0]
        
        # Validate prediction result
        if pd.isna(predicted_price) or predicted_price <= 0:
            logger.error(f"Invalid predicted price: {predicted_price}")
            return None, None
        
        # Calculate confidence based on recent prediction accuracy
        if len(X_test) > 0 and len(y_test) > 0:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mean_actual = y_test.mean()
            if mean_actual > 0:
                confidence = max(0.0, min(1.0, 1.0 - (np.sqrt(mse) / mean_actual)))
            else:
                confidence = 0.5  # Default confidence
        else:
            confidence = 0.5  # Default confidence if no test data
        
        # Get current price with validation
        current_price = df['Close'].iloc[-1]
        if pd.isna(current_price) or current_price <= 0:
            logger.error(f"Invalid current price: {current_price}")
            return None, None
        
        # Log prediction details with safe calculations
        try:
            price_change = predicted_price - current_price
            percent_change = (predicted_price / current_price - 1) * 100
            
            logger.info("=" * 50)
            logger.info("Prediction Summary:")
            logger.info(f"Current Price: ${current_price:.2f}")
            logger.info(f"Predicted Next Day Close: ${predicted_price:.2f}")
            logger.info(f"Predicted Change: ${price_change:.2f} ({percent_change:.2f}%)")
            logger.info(f"Model Confidence: {confidence * 100:.2f}%")
            logger.info("=" * 50)
        except Exception as calc_error:
            logger.warning(f"Error in calculation logging: {calc_error}")
            logger.info(f"Prediction: ${predicted_price:.2f}, Confidence: {confidence:.2f}")
        
        return float(predicted_price), float(confidence)
        
    except Exception as e:
        logger.error(f"Error in predict_next_day_close: {str(e)}\nShape of data: {data.shape}")
        return None, None

def create_strategy_features(data: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
    """Create features based on specific trading strategies."""
    
    df = data.copy()
    
    # Strategy-specific feature combinations
    strategy_features = {
        "Day Trading Calls/Puts": {
            'momentum': (df['RSI_14'] > 50) & (df['MACD'] > df['MACD_Signal']),
            'volume_confirm': df['Volume'] > df['Volume'].rolling(20).mean(),
            'trend_align': df['Close'] > df['EMA_20']
        },
        
        "Swing Trading": {
            'trend_strength': df['ADX'] > 25,
            'ma_alignment': (df['EMA_20'] > df['EMA_50']) & (df['Close'] > df['EMA_20']),
            'momentum_confirm': (df['RSI_14'] > 40) & (df['RSI_14'] < 70)
        },
        
        "Iron Condor": {
            'volatility_high': df['ATR'] > df['ATR'].rolling(20).mean(),
            'range_bound': (df['Close'] > df['BB_lower']) & (df['Close'] < df['BB_upper']),
            'trend_weak': df['ADX'] < 25
        }
    }
    
    # Add strategy-specific features
    if strategy_type in strategy_features:
        for name, condition in strategy_features[strategy_type].items():
            df[f'strategy_{name}'] = condition.astype(int)
    
    return df

def train_strategy_model(data: pd.DataFrame, strategy_type: str) -> Any:
    """Train model with strategy-specific features."""
    
    # Add strategy features
    df = create_strategy_features(data, strategy_type)
    
    # Select features based on strategy
    if "Day Trading" in strategy_type:
        lookback = 5  # Shorter lookback for day trading
        features = ['RSI_14', 'MACD', 'Volume', 'ATR', 'strategy_momentum', 
                   'strategy_volume_confirm', 'strategy_trend_align']
    else:
        lookback = 20  # Longer lookback for swing trading
        features = ['RSI_14', 'ADX', 'ATR', 'BB_upper', 'BB_lower', 
                   'strategy_trend_strength', 'strategy_ma_alignment']
    
    # Create model with optimized parameters
    model = create_model("RandomForest", {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 5
    })
    
    return model, features
