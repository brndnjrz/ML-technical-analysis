"""
Adaptive feature engineering for different timeframes.
Creates features that are appropriate for the specific time interval.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_adaptive_periods(interval: str) -> dict:
    """Return appropriate periods for different timeframes based on trading patterns."""
    
    # Define period mappings based on typical trading patterns
    period_mapping = {
        # Short-term periods for different timeframes
        '1m': {'short': 5, 'medium': 15, 'long': 60, 'volume_window': 10},      # 5min, 15min, 1hour
        '5m': {'short': 3, 'medium': 12, 'long': 48, 'volume_window': 8},      # 15min, 1hour, 4hours  
        '15m': {'short': 4, 'medium': 16, 'long': 64, 'volume_window': 12},    # 1hour, 4hours, 16hours
        '30m': {'short': 2, 'medium': 8, 'long': 32, 'volume_window': 6},      # 1hour, 4hours, 16hours
        '1h': {'short': 4, 'medium': 12, 'long': 24, 'volume_window': 8},      # 4hours, 12hours, 24hours
        '4h': {'short': 3, 'medium': 6, 'long': 18, 'volume_window': 4},       # 12hours, 24hours, 3days
        '1d': {'short': 5, 'medium': 20, 'long': 50, 'volume_window': 15}      # 5days, 20days, 50days
    }
    
    return period_mapping.get(interval, period_mapping['1d'])


def engineer_adaptive_features(data: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Create features that adapt to the timeframe."""
    try:
        df = data.copy()
        periods = get_adaptive_periods(interval)
        
        logger.info(f"Engineering adaptive features for {interval} interval with periods: {periods}")
        
        # Adaptive moving averages
        df['SMA_short'] = ta.sma(df['Close'], length=periods['short']).fillna(df['Close'])
        df['SMA_medium'] = ta.sma(df['Close'], length=periods['medium']).fillna(df['Close'])
        df['SMA_long'] = ta.sma(df['Close'], length=periods['long']).fillna(df['Close'])
        
        # Adaptive EMAs
        df['EMA_short'] = ta.ema(df['Close'], length=periods['short']).fillna(df['Close'])
        df['EMA_medium'] = ta.ema(df['Close'], length=periods['medium']).fillna(df['Close'])
        
        # Trend strength indicators
        df['Trend_Strength_Short'] = (df['Close'] - df['SMA_short']) / df['SMA_short']
        df['Trend_Strength_Medium'] = (df['Close'] - df['SMA_medium']) / df['SMA_medium']
        
        # Price momentum features
        df['Price_Momentum_Short'] = df['Close'].pct_change(periods['short'])
        df['Price_Momentum_Medium'] = df['Close'].pct_change(periods['medium'])
        
        # Adaptive volatility (use shorter windows for shorter timeframes)
        vol_window = max(3, periods['short'])
        df['Volatility_Short'] = df['Close'].pct_change().rolling(window=vol_window).std()
        df['Volatility_Medium'] = df['Close'].pct_change().rolling(window=periods['medium']).std()
        
        # Volatility regime features
        df['Volatility_Regime'] = (df['Volatility_Short'] / df['Volatility_Medium']).fillna(1.0)
        
        # Adaptive momentum indicators  
        rsi_period = max(6, periods['short'] * 2)  # Minimum 6 periods for RSI
        df['RSI_adaptive'] = ta.rsi(df['Close'], length=rsi_period)
        
        # Volume analysis (adapt to timeframe)
        volume_window = periods['volume_window']
        if 'Volume' in df.columns:
            df['Volume_MA'] = ta.sma(df['Volume'], length=volume_window).fillna(df['Volume'])
            df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA']).fillna(1.0)
            df['Volume_Trend'] = df['Volume'].pct_change(periods['short'])
        else:
            df['Volume_MA'] = 0
            df['Volume_Ratio'] = 1.0
            df['Volume_Trend'] = 0
        
        # Market microstructure features (for short timeframes)
        if interval in ['1m', '5m', '15m']:
            # Bid-ask spread proxy using high-low
            df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
            
            # Price impact proxy (only if volume exists)
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                df['Price_Impact'] = (df['Close'] - df['Open']) / (df['Volume'] + 1) * 1e6
            else:
                df['Price_Impact'] = 0
            
            # Intraday momentum
            df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
            
            # High-frequency mean reversion
            df['Mean_Reversion_Short'] = (df['Close'] - df['Close'].rolling(periods['short']).mean()) / df['Close'].rolling(periods['short']).std()
        
        # Advanced trend features for longer timeframes
        if interval in ['1h', '4h', '1d']:
            # ADX trend strength
            adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=periods['medium'])
            if adx_result is not None and not adx_result.empty:
                df['ADX_adaptive'] = adx_result.fillna(25)
            else:
                df['ADX_adaptive'] = 25
            
            # Bollinger Bands
            bb = ta.bbands(df['Close'], length=periods['medium'])
            if bb is not None and not bb.empty:
                bb_middle_col = f'BBM_{periods["medium"]}_2.0'
                bb_upper_col = f'BBU_{periods["medium"]}_2.0'
                bb_lower_col = f'BBL_{periods["medium"]}_2.0'
                
                if bb_middle_col in bb.columns:
                    df['BB_Middle'] = bb[bb_middle_col].fillna(df['Close'])
                    df['BB_Upper'] = bb[bb_upper_col].fillna(df['Close'] * 1.02)
                    df['BB_Lower'] = bb[bb_lower_col].fillna(df['Close'] * 0.98)
                    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                else:
                    df['BB_Position'] = 0.5
                    df['BB_Width'] = 0.04
        
        # Time-based features (important for intraday)
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            df['Hour'] = df.index.hour
            df['Minute'] = df.index.minute
            df['Day_of_Week'] = df.index.dayofweek
            
            # Market session indicators (US market hours)
            df['Market_Open'] = ((df['Hour'] >= 9) & (df['Hour'] < 10)).astype(int)
            df['Market_Close'] = ((df['Hour'] >= 15) & (df['Hour'] < 16)).astype(int)
            df['Lunch_Time'] = ((df['Hour'] >= 12) & (df['Hour'] < 13)).astype(int)
            df['After_Hours'] = ((df['Hour'] < 9) | (df['Hour'] >= 16)).astype(int)
        
        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                if col in ['Price_Momentum_Short', 'Price_Momentum_Medium', 'Volume_Trend', 'Intraday_Return']:
                    df[col] = df[col].fillna(0)
                elif col in ['RSI_adaptive']:
                    df[col] = df[col].fillna(50)
                elif col in ['ADX_adaptive']:
                    df[col] = df[col].fillna(25)
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        logger.info(f"Successfully created {len([col for col in df.columns if col not in data.columns])} adaptive features")
        return df
        
    except Exception as e:
        logger.error(f"Error in adaptive feature engineering: {str(e)}")
        return data


def add_lag_features(data: pd.DataFrame, target_col: str, interval: str) -> pd.DataFrame:
    """Add lag features based on timeframe."""
    try:
        df = data.copy()
        
        # For short timeframes, use more recent lags
        if interval in ['1m', '5m']:
            lags = [1, 2, 3, 5]
        elif interval in ['15m', '30m']:  
            lags = [1, 2, 4, 6]
        elif interval in ['1h', '4h']:
            lags = [1, 2, 6, 12]
        else:  # Daily+
            lags = [1, 2, 5, 10]
        
        # Add lag features for the target
        if target_col in df.columns:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
            # Add rolling statistics of lags
            lag_cols = [f'{target_col}_lag_{lag}' for lag in lags[:3]]
            df[f'{target_col}_lag_mean'] = df[lag_cols].mean(axis=1)
            df[f'{target_col}_lag_std'] = df[lag_cols].std(axis=1).fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding lag features: {str(e)}")
        return data