"""
Preprocessing and scaling utilities for financial features.
Handles different feature types appropriately.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(feature_cols: list, interval: str) -> ColumnTransformer:
    """Create preprocessing pipeline adapted to timeframe and feature types."""
    try:
        # Categorize features by type
        volume_features = [col for col in feature_cols if 'Volume' in col or 'volume' in col.lower()]
        price_features = [col for col in feature_cols if any(x in col for x in ['Close', 'High', 'Low', 'Open', 'SMA', 'EMA', 'BB_'])]
        ratio_features = [col for col in feature_cols if any(x in col for x in ['Ratio', 'Percent', 'RSI', 'Position', 'Strength', 'pct'])]
        momentum_features = [col for col in feature_cols if any(x in col for x in ['Momentum', 'Return', 'Change'])]
        time_features = [col for col in feature_cols if any(x in col for x in ['Hour', 'Minute', 'Day_of_Week', 'Market_', 'After_'])]
        other_features = [col for col in feature_cols if col not in volume_features + price_features + ratio_features + momentum_features + time_features]
        
        preprocessors = []
        
        # Volume features: Log transform then robust scale (due to high variance and skewness)
        if volume_features:
            volume_pipeline = Pipeline([
                ('log_transform', FunctionTransformer(
                    lambda x: np.log1p(np.maximum(x, 0)), 
                    validate=False,
                    feature_names_out='one-to-one'
                )),
                ('robust_scale', RobustScaler())
            ])
            preprocessors.append(('volume', volume_pipeline, volume_features))
        
        # Price features: Robust scaling (handles outliers better than StandardScaler)
        if price_features:
            preprocessors.append(('price', RobustScaler(), price_features))
        
        # Ratio and percentage features: Standard scaling (already normalized)
        if ratio_features:
            preprocessors.append(('ratio', StandardScaler(), ratio_features))
        
        # Momentum features: Robust scaling (can have outliers)
        if momentum_features:
            preprocessors.append(('momentum', RobustScaler(), momentum_features))
        
        # Time features: No scaling needed (categorical/ordinal)
        if time_features:
            preprocessors.append(('time', 'passthrough', time_features))
        
        # Other features: Robust scaling
        if other_features:
            preprocessors.append(('other', RobustScaler(), other_features))
        
        # If no features were categorized, use robust scaling for all
        if not preprocessors:
            preprocessors.append(('all', RobustScaler(), feature_cols))
        
        return ColumnTransformer(preprocessors, remainder='drop')
        
    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {str(e)}")
        # Fallback to simple robust scaling
        return ColumnTransformer([('all', RobustScaler(), feature_cols)], remainder='drop')


def create_target_variable(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Create more sophisticated target variable based on timeframe."""
    try:
        df = data.copy()
        
        # For shorter timeframes, focus on percentage change
        if interval in ['1m', '5m', '15m', '30m']:
            # Use percentage change as target (more stable for short timeframes)
            df['Target'] = df['Close'].pct_change().shift(-1) * 100
            df['Target_Type'] = 'percentage'
            
            # Add volatility adjustment for very short timeframes
            if interval in ['1m', '5m']:
                # Normalize by recent volatility to account for changing market conditions
                volatility = df['Close'].pct_change().rolling(20).std()
                volatility = volatility.fillna(volatility.mean())
                df['Target'] = df['Target'] / (volatility * 100 + 1e-8)  # Add small epsilon to avoid division by zero
                df['Target'] = np.clip(df['Target'], -10, 10)  # Clip extreme values
                
        elif interval in ['1h', '4h']:
            # For medium timeframes, use price change normalized by ATR
            atr = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
            atr = atr.fillna(df['Close'] * 0.02)  # Default to 2% if ATR not available
            
            price_change = df['Close'].shift(-1) - df['Close']
            df['Target'] = price_change / atr
            df['Target_Type'] = 'atr_normalized'
            
        else:
            # For daily and longer timeframes, use absolute price change
            df['Target'] = df['Close'].shift(-1)
            df['Target_Type'] = 'absolute'
        
        # Clip extreme target values to prevent outlier bias
        if df['Target_Type'].iloc[0] != 'absolute':
            target_std = df['Target'].std()
            if not pd.isna(target_std) and target_std > 0:
                df['Target'] = np.clip(df['Target'], 
                                     df['Target'].mean() - 3*target_std,
                                     df['Target'].mean() + 3*target_std)
        
        logger.info(f"Created target variable with type: {df['Target_Type'].iloc[0] if not df.empty else 'unknown'}")
        return df
        
    except Exception as e:
        logger.error(f"Error creating target variable: {str(e)}")
        # Fallback to simple shift
        df['Target'] = df['Close'].shift(-1)
        df['Target_Type'] = 'fallback'
        return df


def handle_missing_values(data: pd.DataFrame, method: str = 'adaptive') -> pd.DataFrame:
    """Handle missing values with appropriate strategies for financial data."""
    try:
        df = data.copy()
        
        if method == 'adaptive':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if df[col].isna().any():
                    # Different strategies for different feature types
                    if 'return' in col.lower() or 'change' in col.lower() or 'momentum' in col.lower():
                        # Returns and changes: fill with 0
                        df[col] = df[col].fillna(0)
                    elif 'rsi' in col.lower():
                        # RSI: fill with neutral value (50)
                        df[col] = df[col].fillna(50)
                    elif 'volume' in col.lower():
                        # Volume: forward fill then median
                        df[col] = df[col].ffill().fillna(df[col].median())
                    elif 'ratio' in col.lower() or 'position' in col.lower():
                        # Ratios: fill with 1 (neutral)
                        df[col] = df[col].fillna(1.0)
                    else:
                        # Others: forward fill then median
                        df[col] = df[col].ffill().fillna(df[col].median())
        
        # Final cleanup: fill any remaining NaN with 0
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        return data.fillna(0)