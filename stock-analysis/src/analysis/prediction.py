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

# Use centralized logging config
logger = logging.getLogger(__name__)

from .market_regime import generate_regime_specific_features, detect_market_regime, volatility_regime_features
from .adaptive_features import engineer_adaptive_features, get_adaptive_periods, add_lag_features
from .preprocessing import create_preprocessing_pipeline, create_target_variable, handle_missing_values
from .adaptive_models import get_adaptive_model_config, calculate_adaptive_ensemble_weights
from .enhanced_validation import get_adaptive_cv_strategy, validate_prediction_performance, select_best_models

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create advanced technical and price action features for prediction."""
    try:
        df = data.copy()
        
        # Price action features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Moving averages and trends with safety checks
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_20'] = df['SMA_20'].ffill().fillna(df['Close'])
        
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_20'] = df['EMA_20'].ffill().fillna(df['Close'])
        
        # Avoid division by zero in SMA_20
        df['Trend_Strength'] = df.apply(lambda x: (x['Close'] - x['SMA_20'])/x['SMA_20'] 
                                     if x['SMA_20'] and x['SMA_20'] > 0 else 0, axis=1)
        
        # Advanced trend features with safety checks
        sma50 = ta.sma(df['Close'], length=50)
        sma50 = sma50.ffill().fillna(df['Close'])
        df['Price_to_SMA50'] = df.apply(lambda x: x['Close'] / sma50.loc[x.name] 
                                      if sma50.loc[x.name] and sma50.loc[x.name] > 0 else 1, axis=1)
        
        sma200 = ta.sma(df['Close'], length=200)
        sma200 = sma200.ffill().fillna(df['Close'])
        df['Price_to_SMA200'] = df.apply(lambda x: x['Close'] / sma200.loc[x.name] 
                                       if sma200.loc[x.name] and sma200.loc[x.name] > 0 else 1, axis=1)
        
        # Safe calculations for crosses
        df['SMA_Cross'] = ((df['SMA_20'] > sma50) * 1).diff()
        df['Golden_Cross'] = ((sma50 > sma200) * 1).diff()
        
        # Momentum indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['RSI_MA'] = ta.sma(df['RSI'], length=5)
        df['RSI_Trend'] = df['RSI'] - df['RSI_MA']
        df['RSI_Divergence'] = ((df['Close'].pct_change(5) > 0) & (df['RSI'].diff(5) < 0)) * -1 + ((df['Close'].pct_change(5) < 0) & (df['RSI'].diff(5) > 0)) * 1
        
        # Volume analysis
        df['Volume_MA'] = ta.sma(df['Volume'], length=20)
        df['Volume_MA'] = df['Volume_MA'].ffill().infer_objects(copy=False)  # Updated for pandas future
        df['Volume_Trend'] = (df['Volume']/df['Volume_MA']).fillna(1.0)  # Default to neutral
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV'] = df['OBV'].ffill().infer_objects(copy=False)  # Updated for pandas future
        
        # Advanced volume features
        df['OBV_Slope'] = ta.slope(df['OBV'], length=5)
        df['Volume_Surge'] = (df['Volume'] > df['Volume_MA'] * 1.5) * 1
        df['Volume_Price_Trend'] = np.sign(df['Returns']) * df['Volume_Trend']
        
        # Volatility indicators
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        # Add safety check for ATR calculation
        df['ATR'] = df['ATR'].fillna(0)
        # Avoid division by zero or None in Close price
        df['ATR_Percent'] = df.apply(lambda x: x['ATR'] / x['Close'] * 100 if x['Close'] and x['Close'] > 0 else 0, axis=1)
        bb = ta.bbands(df['Close'])
        
        # Advanced volatility features with safety checks
        # Initialize with zeros first
        df['BB_Width'] = 0
        df['BB_Position'] = 0.5  # Default to middle of the band
        
        # Only calculate if Bollinger Bands are valid
        if bb is not None and not bb.empty and all(col in bb.columns for col in ['BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0']):
            # Safe calculation of BB Width avoiding division by zero
            mask = (bb['BBM_20_2.0'] > 0)
            df.loc[mask, 'BB_Width'] = (bb.loc[mask, 'BBU_20_2.0'] - bb.loc[mask, 'BBL_20_2.0']) / bb.loc[mask, 'BBM_20_2.0']
            
            # Safe calculation of BB Position
            mask = ((bb['BBU_20_2.0'] - bb['BBL_20_2.0']) > 0)
            df.loc[mask, 'BB_Position'] = (df.loc[mask, 'Close'] - bb.loc[mask, 'BBL_20_2.0']) / (bb.loc[mask, 'BBU_20_2.0'] - bb.loc[mask, 'BBL_20_2.0'])
        
        # Safe calculation for Price Volatility Ratio
        mask = (df['ATR_Percent'] > 0)
        # Initialize with float type explicitly to prevent dtype incompatibility
        df['Price_Volatility_Ratio'] = pd.Series(0.0, index=df.index)
        df.loc[mask, 'Price_Volatility_Ratio'] = df.loc[mask, 'Returns'].abs() / df.loc[mask, 'ATR_Percent']
        
        # Market regime features
        df['Trend_Regime'] = ((df['Close'] > df['SMA_20']) & (df['SMA_20'] > ta.sma(df['Close'], length=50))) * 1 + \
                           ((df['Close'] < df['SMA_20']) & (df['SMA_20'] < ta.sma(df['Close'], length=50))) * -1
        df['Volatility_Regime'] = (df['ATR_Percent'] > df['ATR_Percent'].rolling(window=50).mean()) * 1
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
        
        # Add market regime features - with safety checks
        try:
            market_regime = detect_market_regime(df)
            df = generate_regime_specific_features(df, market_regime)
        except Exception as regime_error:
            logger.warning(f"Error adding market regime features: {str(regime_error)}")
            # Continue without these features
        
        # Add volatility regime features - with safety checks
        try:
            df = volatility_regime_features(df)
        except Exception as vol_error:
            logger.warning(f"Error adding volatility features: {str(vol_error)}")
            # Continue without these features
        
        # Log the regime detection results
        logger.info(f"Market regime detected: {market_regime}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in engineer_features: {str(e)}")
        raise

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    random_forest = {
        'n_estimators': 500,  # Increased for better accuracy
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',  # Better generalization
        'bootstrap': True,
        'oob_score': True,  # Out-of-bag scoring for better validation
        'n_jobs': -1,  # Use all cores
        'random_state': 42
    }
    
    xgboost = {
        'n_estimators': 300,  # Increased for better accuracy
        'learning_rate': 0.03,  # Decreased for better generalization
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'gamma': 1,  # Regularization parameter
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'scale_pos_weight': 1,
        'random_state': 42
    }
    
    catboost = {
        'iterations': 500,  # Increased from 200
        'learning_rate': 0.03,  # Decreased for better generalization
        'depth': 8,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU',
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0  # Controls randomness in Bayesian bootstrap
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
        
        # Initialize default values
        eps = 0
        pe_ratio = 0
        revenue_growth = 0
        profit_margin = 0
        
        # Try to use fast_info (safer, less prone to timeouts)
        try:
            fast_info = yf_ticker.fast_info
            
            # Safe attribute access with fallback to default values
            if hasattr(fast_info, "trailing_pe"):
                pe_ratio = fast_info.trailing_pe if fast_info.trailing_pe is not None else 0
        except Exception as fast_info_error:
            print(f"Error accessing fast_info: {fast_info_error}")
        
        # Try to get more detailed info (more likely to fail with timeouts)
        try:
            # Get minimal info safely
            info = {}
            with st.spinner("Fetching additional fundamental data..."):
                try:
                    # Basic info request - don't use private methods anymore
                    basic_info = yf_ticker.info
                    
                    # Check if we got a dictionary back
                    if isinstance(basic_info, dict):
                        info = basic_info
                    else:
                        print(f"Unexpected info type: {type(basic_info)}")
                except Exception as info_error:
                    print(f"Error getting basic info: {info_error}")
            
            # Safe dictionary access
            if isinstance(info, dict):
                eps = info.get("trailingEps", eps)
                if pe_ratio == 0:  # Only use if we didn't get it from fast_info
                    pe_ratio = info.get("trailingPE", pe_ratio)
                revenue_growth = info.get("revenueGrowth", revenue_growth)
                profit_margin = info.get("profitMargins", profit_margin)
        except Exception as e:
            print(f"Could not fetch detailed fundamentals: {e}")
        
        # Return metrics with safe values
        return {
            "EPS": eps,
            "P/E Ratio": pe_ratio,
            "Revenue Growth": revenue_growth,
            "Profit Margin": profit_margin,
        }
    except Exception as e:
        st.warning(f"Could not fetch fundamentals: {e}")
        return {}

def predict_next_period_close(data: pd.DataFrame, fundamentals: dict, selected_indicators: list, interval: str = '1d') -> Tuple[float, float]:
    """Predicts the next period's closing price using an ensemble of models for better accuracy.
    The 'period' is determined by the 'interval' parameter (e.g., '15m', '1h', '1d')."""
    try:
        # Input validation
        if data is None or data.empty:
            logger.error("Input data is None or empty")
            return None, None
        
        if 'Close' not in data.columns:
            logger.error("'Close' column not found in data")
            return None, None
        
        # Check if we have enough data points
        if len(data) < 30:  # Increased minimum data points for better accuracy
            logger.error(f"Insufficient data points: {len(data)}. Need at least 30 for reliable predictions.")
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
        
        # Engineer features with enhanced adaptive approach
        try:
            logger.info(f"Starting adaptive feature engineering for {interval} interval")
            df = engineer_adaptive_features(df, interval)
            
            # Add original features as fallback if adaptive fails
            if len([col for col in df.columns if col not in data.columns]) < 5:
                logger.warning("Adaptive feature engineering produced few features, adding fallback features")
                df = engineer_features(df)
                
        except Exception as feature_error:
            logger.warning(f"Adaptive feature engineering failed: {feature_error}. Using basic features.")
            try:
                df = engineer_features(df)
            except Exception as basic_error:
                logger.warning(f"Basic feature engineering also failed: {basic_error}. Using fallback.")
                # Fallback to basic features if engineering fails
                df['Returns'] = df['Close'].pct_change().fillna(0)
                periods = get_adaptive_periods(interval)
                df['SMA_short'] = df['Close'].rolling(periods['short']).mean().fillna(df['Close'])
                df['SMA_medium'] = df['Close'].rolling(periods['medium']).mean().fillna(df['Close'])
                if 'Volume' in df.columns:
                    df['Volume_MA'] = df['Volume'].rolling(periods['volume_window']).mean().fillna(df['Volume'])
                else:
                    df['Volume_MA'] = df['Close'] * 0
        
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
        
        # Create adaptive target variable
        df = create_target_variable(df, interval)
        target_type = df.get('Target_Type', pd.Series(['absolute'])).iloc[0] if not df.empty else 'absolute'
        logger.info(f"Using target type: {target_type}")
        
        # Add lag features for the target
        if 'Target' in df.columns:
            df = add_lag_features(df, 'Returns', interval)
        
        # Filter out any completely empty feature columns
        valid_feature_cols = []
        for col in feature_cols:
            if col in df.columns and not df[col].isna().all():
                valid_feature_cols.append(col)
        
        if len(valid_feature_cols) == 0:
            logger.error("No valid feature columns with data")
            return None, None
            
        feature_cols = valid_feature_cols
        
        # Filter out string/object columns that can't be used in models
        numeric_feature_cols = []
        for col in feature_cols:
            try:
                # Check if the column is a Series and if it has a dtype property
                if isinstance(df[col], pd.Series) and hasattr(df[col], 'dtype'):
                    if df[col].dtype.kind in 'biufc':  # boolean, integer, unsigned int, float, complex
                        numeric_feature_cols.append(col)
                    elif col == 'Market_Regime':
                        # One-hot encode the Market_Regime column
                        try:
                            logger.info(f"One-hot encoding Market_Regime: {df['Market_Regime'].unique()}")
                            regime_dummies = pd.get_dummies(df['Market_Regime'], prefix='Regime')
                            # Ensure dummy columns are properly added as Series, not DataFrames
                            for dummy_col in regime_dummies.columns:
                                # Check if the column already exists to avoid duplicates
                                if dummy_col not in df.columns:
                                    # Convert to Series explicitly
                                    dummy_series = pd.Series(regime_dummies[dummy_col].values, 
                                                           index=df.index, 
                                                           name=dummy_col, 
                                                           dtype='int64')
                                    df[dummy_col] = dummy_series
                                    # Verify the column is a Series before adding to features
                                    if isinstance(df[dummy_col], pd.Series):
                                        numeric_feature_cols.append(dummy_col)
                                        logger.info(f"Added dummy column {dummy_col} as Series")
                                    else:
                                        logger.warning(f"Failed to create Series for dummy column {dummy_col}")
                                else:
                                    logger.info(f"Dummy column {dummy_col} already exists, skipping")
                        except Exception as e:
                            logger.warning(f"Could not one-hot encode Market_Regime: {e}")
                    else:
                        logger.warning(f"Skipping non-numeric column: {col} with dtype {df[col].dtype}")
                else:
                    # If it's a DataFrame or something else, handle accordingly
                    logger.warning(f"Skipping column {col} which is not a Series: {type(df[col])}")
            except Exception as e:
                logger.warning(f"Error processing column {col}: {e}")
                # Don't try to access df[col].dtype here, as it might be the source of the exception
                
        feature_cols = numeric_feature_cols
        logger.info(f"Using {len(feature_cols)} numeric feature columns for prediction")
        
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
        
        # Create and train model with adaptive configuration and enhanced validation
        try:
            # Get adaptive model configurations
            model_configs = get_adaptive_model_config(interval)
            
            # Set up adaptive cross-validation
            cv_strategy = get_adaptive_cv_strategy(len(X_train), interval)
            
            # Detect current market regime for adaptive modeling
            last_n_rows = df.iloc[-30:] if len(df) >= 30 else df
            current_regime = detect_market_regime(last_n_rows)
            logger.info(f"Current market regime detected: {current_regime}")
            
            # Apply preprocessing
            try:
                preprocessor = create_preprocessing_pipeline(feature_cols, interval)
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test) if len(X_test) > 0 else None
                
                # Convert back to DataFrame with proper column names
                feature_names = preprocessor.get_feature_names_out()
                X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
                if X_test_processed is not None:
                    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
                    
            except Exception as prep_error:
                logger.warning(f"Preprocessing failed: {prep_error}. Using original features.")
                X_train_processed = X_train
                X_test_processed = X_test
            
            # Create models with adaptive configurations
            models = {}
            for model_name, config in model_configs.items():
                try:
                    if model_name == 'RandomForest':
                        models[model_name] = RandomForestRegressor(**config)
                    elif model_name == 'XGBoost':
                        models[model_name] = XGBRegressor(**config)
                    elif model_name == 'CatBoost':
                        models[model_name] = CatBoostRegressor(**config)
                except Exception as model_error:
                    logger.warning(f"Error creating {model_name}: {model_error}")
                    continue
            
            # Enhanced cross-validation
            validation_results = validate_prediction_performance(
                models, X_train_processed, y_train, cv_strategy, interval
            )
            
            # Select best models and calculate adaptive weights
            best_model_names = select_best_models(validation_results, interval)
            best_models = {name: models[name] for name in best_model_names if name in models}
            
            if not best_models:
                logger.warning("No valid models found, using fallback")
                best_models = {'RandomForest': RandomForestRegressor(random_state=42)}
                validation_results = {'RandomForest': [{'rmse': 1.0}]}
            
            # Calculate adaptive ensemble weights
            model_weights = calculate_adaptive_ensemble_weights(validation_results, {}, interval)
            
            # Apply regime-specific model weight adjustments
            if current_regime == 'volatile_bullish':
                # In volatile bullish markets, favor models that handle noise well
                if 'CatBoost' in model_weights:
                    model_weights['CatBoost'] *= 1.2
                if 'XGBoost' in model_weights:
                    model_weights['XGBoost'] *= 1.1
            elif current_regime == 'volatile_bearish':
                # In volatile bearish markets, give more weight to robust models
                if 'CatBoost' in model_weights:
                    model_weights['CatBoost'] *= 1.5
            elif current_regime == 'range_bound':
                # In range-bound markets, favor RandomForest
                if 'RandomForest' in model_weights:
                    model_weights['RandomForest'] *= 1.4
            
            # Normalize weights
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                model_weights = {name: weight/total_weight for name, weight in model_weights.items()}
            
            logger.info(f"Final model weights for {current_regime} regime: {model_weights}")
            
            # Train final models on full training data
            for name, model in best_models.items():
                try:
                    model.fit(X_train_processed, y_train)
                except Exception as train_error:
                    logger.warning(f"Training failed for {name}: {train_error}")
                    if name in model_weights:
                        del model_weights[name]
            
            # Get last row for prediction
            last_row = X.iloc[[-1]]
            try:
                last_row_processed = preprocessor.transform(last_row)
                last_row_processed = pd.DataFrame(last_row_processed, columns=feature_names, index=last_row.index)
            except:
                last_row_processed = last_row[feature_cols]
            
            # Make predictions with each model
            predictions = {}
            for name, model in best_models.items():
                try:
                    pred = model.predict(last_row_processed)[0]
                    predictions[name] = pred
                    logger.info(f"Model {name} prediction: {pred:.4f}")
                except Exception as pred_error:
                    logger.warning(f"Model {name} prediction failed: {pred_error}")
                    if name in model_weights:
                        del model_weights[name]
            
            # Filter out failed predictions and recalculate weights
            valid_predictions = {name: pred for name, pred in predictions.items() if name in model_weights}
            if not valid_predictions:
                logger.error("All model predictions failed")
                return None, None
            
            # Renormalize weights for valid predictions
            valid_weight_sum = sum(model_weights[name] for name in valid_predictions)
            if valid_weight_sum > 0:
                for name in valid_predictions:
                    model_weights[name] /= valid_weight_sum
            
            # Ensemble prediction (weighted average)
            predicted_price = sum(valid_predictions[name] * model_weights[name] for name in valid_predictions)
            
            # Convert prediction based on target type
            current_price = df['Close'].iloc[-1]
            if target_type == 'percentage':
                # Convert percentage change back to price
                predicted_price = current_price * (1 + predicted_price / 100)
            elif target_type == 'atr_normalized':
                # Convert ATR-normalized change back to price
                atr = df.get('ATR', pd.Series([current_price * 0.02])).iloc[-1]
                predicted_price = current_price + (predicted_price * atr)
            # For absolute target, predicted_price is already the target price
            
            # Store the best performing model for reference
            best_model_name = max(model_weights, key=model_weights.get) if model_weights else 'Unknown'
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Individual predictions: {valid_predictions}")
            logger.info(f"Ensemble prediction: {predicted_price:.4f}")
            
        except Exception as model_error:
            logger.error(f"Adaptive model training failed: {model_error}")
            return None, None
        
        # Validate prediction result
        if pd.isna(predicted_price) or predicted_price <= 0:
            logger.error(f"Invalid predicted price: {predicted_price}")
            return None, None
        
        # Calculate confidence based on recent prediction accuracy with adaptive regime adjustment
        if len(X_test) > 0 and len(y_test) > 0 and X_test_processed is not None:
            # Get ensemble predictions for test set
            test_predictions = {}
            for name, model in best_models.items():
                if name in model_weights:
                    try:
                        test_predictions[name] = model.predict(X_test_processed)
                    except Exception as test_pred_error:
                        logger.warning(f"Test prediction failed for {name}: {test_pred_error}")
                        continue
            
            if test_predictions:
                # Calculate weighted ensemble predictions for test set
                ensemble_test_preds = np.zeros_like(y_test)
                for name in test_predictions:
                    if name in model_weights:
                        ensemble_test_preds += test_predictions[name] * model_weights[name]
                
                # Calculate timeframe-specific confidence metrics
                if interval in ['1m', '5m', '15m', '30m']:
                    # For short timeframes, use directional accuracy as primary confidence metric
                    directional_accuracy = np.mean(np.sign(ensemble_test_preds) == np.sign(y_test))
                    base_confidence = directional_accuracy
                    
                    # Add hit rate component
                    threshold = np.std(y_test) * 0.5
                    hit_rate = np.mean(np.abs(ensemble_test_preds - y_test) <= threshold)
                    base_confidence = (directional_accuracy * 0.7 + hit_rate * 0.3)
                    
                else:
                    # For longer timeframes, use normalized RMSE
                    mse = mean_squared_error(y_test, ensemble_test_preds)
                    mean_actual = np.abs(y_test).mean()
                    
                    if mean_actual > 0:
                        base_confidence = max(0.0, min(1.0, 1.0 - (np.sqrt(mse) / mean_actual)))
                    else:
                        base_confidence = 0.5  # Default confidence
                
                # Adjust confidence based on market regime and timeframe
                regime_confidence_adjustments = {
                    'trending_bullish': 0.08 if interval in ['1h', '4h', '1d'] else 0.05,
                    'trending_bearish': 0.05 if interval in ['1h', '4h', '1d'] else 0.03,
                    'volatile_bullish': -0.10 if interval in ['1m', '5m'] else -0.05,
                    'volatile_bearish': -0.15 if interval in ['1m', '5m'] else -0.08,
                    'range_bound': 0.0
                }
                
                # Apply regime-specific confidence adjustment
                regime_adjustment = regime_confidence_adjustments.get(current_regime, 0.0)
                confidence = max(0.1, min(0.95, base_confidence + regime_adjustment))
                
                logger.info(f"Base confidence: {base_confidence:.2f}, Regime adjustment: {regime_adjustment:.2f}")
            else:
                confidence = 0.5  # Default if no test predictions
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
            logger.info(f"Predicted Next {interval} Close: ${predicted_price:.2f}")
            logger.info(f"Predicted Change: ${price_change:.2f} ({percent_change:.2f}%)")
            logger.info(f"Model Confidence: {confidence * 100:.2f}%")
            logger.info(f"Market Regime: {current_regime}")
            logger.info(f"Model Weights: {', '.join([f'{name}: {weight:.2f}' for name, weight in model_weights.items()])}")
            logger.info("=" * 50)
        except Exception as calc_error:
            logger.warning(f"Error in calculation logging: {calc_error}")
            logger.info(f"Prediction: ${predicted_price:.2f}, Confidence: {confidence:.2f}")
        
        return float(predicted_price), float(confidence)
        
    except ValueError as ve:
        if "could not convert string to float" in str(ve):
            logger.error(f"Model training failed due to non-numeric data: {str(ve)}")
            logger.error("This error usually happens when categorical data like 'Market_Regime' is not properly encoded.")
            
            # Try to find which column contains this value
            import re
            error_str = str(ve)
            match = re.search(r"could not convert string to float: '([^']*)'", error_str)
            if match:
                problem_value = match.group(1)
                logger.error(f"Problematic value: '{problem_value}'")
                
                # Try to find which column contains this value
                for col in feature_cols:
                    if col in df.columns and df[col].astype(str).eq(problem_value).any():
                        logger.error(f"Problematic column found: {col}")
                        # Remove this column from features to avoid the error in the future
                        if col in feature_cols:
                            feature_cols.remove(col)
                            logger.info(f"Removed '{col}' from feature columns to prevent errors")
        return None, None
                
    except Exception as e:
        logger.error(f"Error in predict_next_period_close: {str(e)}\nShape of data: {data.shape}")
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
        
        "Iron Condors": {
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
