# =========================
# Imports and Setup
# =========================
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import inspect
from typing import Optional, List, Dict, Any, Callable, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def safe_calculate(df: pd.DataFrame, func: Callable, indicator_name: str) -> Union[pd.DataFrame, pd.Series]:
    """Safely calculate an indicator using the provided function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        func (Callable): Function to calculate the indicator
        indicator_name (str): Name of the indicator for logging
        
    Returns:
        Union[pd.DataFrame, pd.Series]: Calculated indicator or empty DataFrame/Series
    """
    try:
        if df.empty:
            logger.error(f"[{indicator_name}] Input DataFrame is empty")
            if isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            return pd.Series()
            
        # Get required columns based on the function signature
        required_cols = inspect.getfullargspec(func).args
        
        # Check if required columns exist
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"[{indicator_name}] Missing required columns: {missing}")
            if isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            return pd.Series()
            
        # Remove any NaN rows that could cause issues
        clean_df = df.dropna(subset=required_cols)
        if clean_df.empty:
            logger.error(f"[{indicator_name}] No valid data after dropping NaN values")
            if isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            return pd.Series()
            
        try:
            # Calculate indicator
            result = func(clean_df)
            return result if result is not None else pd.Series()
            
        except Exception as e:
            logger.error(f"[{indicator_name}] Error calculating indicator: {str(e)}")
            if isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            return pd.Series()
            
    except Exception as e:
        logger.error(f"[{indicator_name}] Error in data preparation: {str(e)}")
        if isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        return pd.Series()

def get_supported_timeframes(data: pd.DataFrame) -> pd.DataFrame:
    """Get OHLCV data resampled to different timeframes.
    
    Args:
        data (pd.DataFrame): Original OHLCV DataFrame
        
    Returns:
        dict: Dictionary with resampled DataFrames
    """
    timeframes = {
        "1m": "1min",
        "5m": "5min", 
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D"
    }
    
    try:
        if data.empty:
            logger.error("Input DataFrame is empty")
            return pd.DataFrame()
            
        # Check required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in data.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
            
        # Remove rows with NaN values
        clean_data = data.dropna(subset=required)
        if clean_data.empty:
            logger.error("No valid data after dropping NaN values")
            return pd.DataFrame()
            
        try:
            # Calculate base metrics for each timeframe
            result = {}
            for tf_key, tf_value in timeframes.items():
                result[tf_key] = calculate_indicators(data, timeframe=tf_key)
            return result
            
        except Exception as e:
            logger.error(f"Error calculating timeframe metrics: {str(e)}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(data: pd.DataFrame, 
                       timeframe: str = "1d",
                       strategy_type: Optional[str] = None,
                       selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate technical indicators for market analysis.
    
    Args:
        data: DataFrame with OHLCV data
        timeframe: Data timeframe (e.g., "1d", "1h")
        strategy_type: Type of trading strategy
        selected_indicators: List of indicators to calculate
        
    Returns:
        DataFrame with calculated indicators
    """
    
    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.error("Invalid or empty input data")
        return pd.DataFrame()
    
    try:
        # Make a copy of input data
        df = data.copy()
        
        # Verify required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            logger.error("Missing required OHLCV columns")
            return pd.DataFrame()
            
        # Set default indicators if none provided
        if not selected_indicators:
            selected_indicators = ["RSI", "MACD", "BB", "ADX", "ATR"]
            
        # Basic calculations with error handling
        try:
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate volatility
            df['volatility'] = (df['returns']
                              .rolling(window=21)
                              .std()
                              .replace([np.inf, -np.inf], np.nan)
                              .ffill()
                              .fillna(0) * np.sqrt(252))
                              
            # RSI
            df['RSI'] = ta.rsi(df['Close'], length=14).fillna(50)
            df[f'RSI_{timeframe}'] = df['RSI']
            
            # MACD
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9'].fillna(0)
            df['MACD_Signal'] = macd['MACDs_12_26_9'].fillna(0)
            df[f'MACD_{timeframe}'] = df['MACD']
            df[f'MACD_Signal_{timeframe}'] = df['MACD_Signal']
            
            # ADX
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14'].fillna(20)
            df[f'ADX_{timeframe}'] = df['ADX']
            
            # OBV
            df['OBV'] = ta.obv(df['Close'], df['Volume']).fillna(0)
            df[f'OBV_{timeframe}'] = df['OBV']
            
            # ATR
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close']).fillna(0)
            df[f'ATR_{timeframe}'] = df['ATR']
            
            # Clean up any remaining NaN values
            df = df.ffill().bfill().fillna(0)

            # --- Ensure all required columns exist, even if not calculated ---
            required_cols = [
                'EMA_20', 'EMA_50', 'OBV', 'ATR', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ADX',
                'BB_upper', 'BB_middle', 'BB_lower', 'VWAP', 'Volume'
            ]
            for col in required_cols:
                if col not in df.columns:
                    if col == 'Volume' and 'Volume' in data.columns:
                        df[col] = data['Volume']
                    else:
                        df[col] = 0

            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return data  # Return original data if calculation fails
            
    except Exception as e:
        logger.error(f"Error in indicator calculation setup: {str(e)}")
        return pd.DataFrame()
# =========================
# Support & Resistance Detection
# =========================
def detect_support_resistance(data, method="quick", window=20, tolerance=0.01):
    """
    Detect support and resistance zones using local minima and maxima in price history.

    Args:
        data (pd.DataFrame): Stock price data
        method (str): Method to use ("quick" or "advanced")
        window (int): Window size for detecting levels
        tolerance (float): Tolerance for grouping similar levels

    Returns:
        dict: {'support': [...], 'resistance': [...]}
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
    df = data.copy()
    supports = []
    resistances = []

    window = max(window, 30) if method == "advanced" else max(window, 10)
    if len(df) < window * 2:
        print(f"[Warning] Not enough data for support/resistance detection. Need at least {window*2} candles, got {len(df)}")
        return {"support": [], "resistance": []}

    for i in range(window, len(df) - window):
        local_lows = df['Low'].iloc[i - window:i + window]
        if df['Low'].iloc[i] == local_lows.min():
            level = df['Low'].iloc[i]
            if not any(abs(level - s) / s < tolerance for s in supports):
                supports.append(level)
        local_highs = df['High'].iloc[i - window:i + window]
        if df['High'].iloc[i] == local_highs.max():
            level = df['High'].iloc[i]
            if not any(abs(level - r) / r < tolerance for r in resistances):
                resistances.append(level)

    supports = sorted(supports)
    resistances = sorted(resistances)
    return {
        "support": supports[-3:] if supports else [],
        "resistance": resistances[-3:] if resistances else []
    }

# =========================
# IV Metrics Calculation
# =========================
def calculate_iv_metrics(ticker, data):
    """
    Calculate implied volatility metrics from historical data.
    """
    try:
        if 'volatility' not in data.columns:
            data['returns'] = data['Close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=21).std() * (252 ** 0.5)
        current_iv = data['volatility'].iloc[-1] * 100
        iv_series = data['volatility'].dropna()
        iv_rank = (iv_series.rank(pct=True).iloc[-1]) * 100
        return {
            'iv_rank': iv_rank,
            'iv_percentile': iv_rank,
            'hv_30': current_iv,
            'vix': 20.0  # Placeholder
        }
    except Exception as e:
        print(f"[Warning] Could not calculate IV metrics: {e}")
        return {
            'iv_rank': 0,
            'iv_percentile': 0,
            'hv_30': 0,
            'vix': 20.0
        }

# =========================
# Options Flow (Placeholder)
# =========================
def get_options_flow(ticker):
    """
    Get options flow data (placeholder).
    """
    # print(f"[Info] Options flow data not available for {ticker}")
    logging.info(f"Options flow data not available for {ticker}")
    return None
    # Default indicators if none selected
    if selected_indicators is None:
        selected_indicators = ["RSI", "MACD", "SMA", "EMA", "VWAP", "Bollinger Bands", "ADX", "Stochastic", "OBV", "ATR"]
    selected_str = str(selected_indicators).lower()

    try:
        # Implied Volatility (IV) with safety checks
        data['returns'] = data['Close'].pct_change()
        data['returns'] = data['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        volatility = data['returns'].rolling(window=21).std()
        volatility = volatility.replace([np.inf, -np.inf], np.nan)
        data['volatility'] = (volatility.fillna(method='ffill')
                            .fillna(method='bfill')
                            .fillna(0) * (252 ** 0.5))
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        data['returns'] = 0
        data['volatility'] = 0
        
    try:
        # RSI
        if any(indicator.lower() in selected_str for indicator in ['rsi']):
            data['RSI_14'] = ta.rsi(data['Close'], length=14).ffill().fillna(50)
            data['RSI_21'] = ta.rsi(data['Close'], length=21).ffill().fillna(50)
            data[f'RSI_{timeframe}'] = data['RSI_14']
            data['RSI'] = data['RSI_14']  # Default RSI
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        data['RSI_14'] = 50
        data['RSI_21'] = 50
        data[f'RSI_{timeframe}'] = 50
        data['RSI'] = 50
    for col in ['RSI_14', 'RSI_21', f'RSI_{timeframe}', 'RSI']:
        data[col] = data[col].ffill().fillna(50)

    # MACD
    try:
        macd = ta.macd(data['Close'])
        data['MACD'] = macd['MACD_12_26_9'].ffill().fillna(0)
        data['MACD_Signal'] = macd['MACDs_12_26_9'].ffill().fillna(0)
        data['MACD_Hist'] = macd['MACDh_12_26_9'].ffill().fillna(0)
        
        # Store timeframe specific versions
        data[f'MACD_{timeframe}'] = data['MACD']
        data[f'MACD_Signal_{timeframe}'] = data['MACD_Signal']
        data[f'MACD_Hist_{timeframe}'] = data['MACD_Hist']
        # Store additional formats that might be needed
        data['MACD_Line'] = data['MACD']
        data['Signal_Line'] = data['MACD_Signal']
        data['MACD_Histogram'] = data['MACD_Hist']
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        for col in ['MACD', 'MACD_Signal', 'MACD_Hist', 
                   f'MACD_{timeframe}', f'MACD_Signal_{timeframe}', f'MACD_Hist_{timeframe}',
                   'MACD_Line', 'Signal_Line', 'MACD_Histogram']:
            data[col] = 0
    
    macd_cols = ['MACD', 'MACD_Signal', 'MACD_Hist', 
                 f'MACD_{timeframe}', f'MACD_Signal_{timeframe}', f'MACD_Hist_{timeframe}',
                 'MACD_Line', 'Signal_Line', 'MACD_Histogram']
    for col in macd_cols:
        if col in data.columns:
            data[col] = data[col].ffill().fillna(0)

    # SMA
    try:
        data['SMA_20'] = data['Close'].rolling(window=20).mean().ffill()
        data['SMA_50'] = data['Close'].rolling(window=50).mean().ffill()
        
        # Store timeframe specific versions
        data[f'SMA_20_{timeframe}'] = data['SMA_20']
        data[f'SMA_50_{timeframe}'] = data['SMA_50']
        # Add raw SMA columns without timeframe suffix
        data['SMA20'] = data['SMA_20']
        data['SMA50'] = data['SMA_50']
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}")
        for col in ['SMA_20', 'SMA_50', f'SMA_20_{timeframe}', f'SMA_50_{timeframe}', 'SMA20', 'SMA50']:
            data[col] = data['Close']
    
    sma_cols = ['SMA_20', 'SMA_50', f'SMA_20_{timeframe}', f'SMA_50_{timeframe}', 'SMA20', 'SMA50']
    for col in sma_cols:
        data[col] = data[col].ffill().fillna(data['Close'])

    # EMA
    try:
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean().ffill()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean().ffill()
        
        # Store timeframe specific versions
        data[f'EMA_20_{timeframe}'] = data['EMA_20']
        data[f'EMA_50_{timeframe}'] = data['EMA_50']
        
        ema_cols = ['EMA_20', 'EMA_50', f'EMA_20_{timeframe}', f'EMA_50_{timeframe}']
        for col in ema_cols:
            data[col] = data[col].ffill().fillna(data['Close'])
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        for col in ['EMA_20', 'EMA_50', f'EMA_20_{timeframe}', f'EMA_50_{timeframe}']:
            data[col] = data['Close']

    # VWAP
    try:
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data["Volume"].cumsum()
        data['VWAP'] = data['VWAP'].ffill().fillna(data['Close'])
        data[f'VWAP_{timeframe}'] = data['VWAP']
    except Exception as e:
        logger.warning(f"Error calculating VWAP: {str(e)}")
        data['VWAP'] = data['Close']
        data[f'VWAP_{timeframe}'] = data['Close']

    # Bollinger Bands
    try:
        bb = ta.bbands(data['Close'], length=20, std=2)
        data[f'BB_middle_{timeframe}'] = bb['BBM_20_2.0'].ffill().fillna(data['Close'])
        std = data['Close'].rolling(window=20).std().ffill().fillna(data['Close'].std())
        
        data[f'BB_upper_{timeframe}'] = bb['BBU_20_2.0'].fillna(data[f'BB_middle_{timeframe}'] + 2 * std)
        data[f'BB_lower_{timeframe}'] = bb['BBL_20_2.0'].fillna(data[f'BB_middle_{timeframe}'] - 2 * std)
        
        # Store standard versions without timeframe
        data['BB_upper'] = data[f'BB_upper_{timeframe}']
        data['BB_lower'] = data[f'BB_lower_{timeframe}']
        data['BB_middle'] = data[f'BB_middle_{timeframe}']
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        data[f'BB_middle_{timeframe}'] = data['Close']
        data[f'BB_upper_{timeframe}'] = data['Close']
        data[f'BB_lower_{timeframe}'] = data['Close']
        data['BB_upper'] = data['Close']
        data['BB_lower'] = data['Close']
        data['BB_middle'] = data['Close']
    
    bb_cols = ['BB_upper', 'BB_lower', 'BB_middle',
               f'BB_upper_{timeframe}', f'BB_lower_{timeframe}', f'BB_middle_{timeframe}']
    for col in bb_cols:
        data[col] = data[col].ffill()

    # ADX
    try:
        adx_result = ta.adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx_result['ADX_14'].ffill().fillna(25)
        data[f'ADX_{timeframe}'] = data['ADX']
        
        for col in ['ADX', f'ADX_{timeframe}']:
            data[col] = data[col].ffill().fillna(25)
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        for col in ['ADX', f'ADX_{timeframe}']:
            data[col] = 25

    # Stochastic
    try:
        if any(indicator.lower() in selected_str for indicator in ['stoch']):
            stoch = ta.stoch(data['High'], data['Low'], data['Close'])
            data[f'STOCH_%K_{timeframe}'] = stoch['STOCHk_14_3_3']
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {str(e)}")
        data[f'STOCH_%K_{timeframe}'] = 50
        data[f'STOCH_%D_{timeframe}'] = stoch['STOCHd_14_3_3']

    # OBV
    try:
        if any(indicator.lower() in selected_str for indicator in ['obv']):
            data['OBV'] = ta.obv(data['Close'], data['Volume'])
            data[f'OBV_{timeframe}'] = data['OBV']
    except Exception as e:
        logger.error(f"Error calculating OBV: {str(e)}")
        data['OBV'] = 0
        data[f'OBV_{timeframe}'] = 0

    # ATR (always calculate)
    try:
        atr = ta.atr(data['High'], data['Low'], data['Close'])
        initial_atr = (data['High'].iloc[0] - data['Low'].iloc[0]) if not atr.empty else 0
        data['ATR'] = atr.ffill().fillna(initial_atr)
        data[f'ATR_{timeframe}'] = data['ATR']
        
        for col in ['ATR', f'ATR_{timeframe}']:
            if col in data.columns:
                data[col] = data[col].ffill().fillna(initial_atr)
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        initial_atr = data['High'].iloc[0] - data['Low'].iloc[0]
        for col in ['ATR', f'ATR_{timeframe}']:
            data[col] = initial_atr

    # Verify indicator calculations
    print("\nIndicator Calculation Summary:")
    latest_data = data.iloc[-1]
    
    # Group indicators for better readability
    indicator_groups = {
        'Momentum Indicators': ['RSI_21', 'RSI_14', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'],
        'Moving Averages': ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50'],
        'Trend Indicators': ['ADX', 'VWAP'],
        'Volatility Indicators': ['ATR', 'BB_upper_' + timeframe, 'BB_lower_' + timeframe]
    }
    
    print("\n=== Technical Indicator Values ===")
    for group_name, indicators in indicator_groups.items():
        print(f"\n{group_name}:")
        for indicator in indicators:
            if indicator in data.columns:
                try:
                    value = latest_data[indicator]
                    if pd.isna(value):
                        print(f"  {indicator}: N/A (NaN value)")
                    else:
                        print(f"  {indicator}: {value:.4f}")
                except Exception as e:
                    print(f"  {indicator}: Error calculating ({str(e)})")
            else:
                print(f"  {indicator}: Not calculated")
    
    # Verify data quality
    null_columns = data.columns[data.isnull().any()].tolist()
    if null_columns:
        print("\nWarning: Found null values in columns:")
        for col in null_columns:
            null_count = data[col].isnull().sum()
            print(f"  {col}: {null_count} null values")
    
    # Print all available columns for debugging
    print("\nAvailable Columns:")
    print(", ".join(sorted(data.columns)))
    
    return data

# =========================
# Support & Resistance Detection
# =========================
def detect_support_resistance(data, method="quick", window=20, tolerance=0.01):
    """
    Detect support and resistance zones using local minima and maxima in price history.

    Args:
        data (pd.DataFrame): Stock price data
        method (str): Method to use ("quick" or "advanced")
        window (int): Window size for detecting levels
        tolerance (float): Tolerance for grouping similar levels

    Returns:
        dict: {'support': [...], 'resistance': [...]}
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
    df = data.copy()
    supports = []
    resistances = []

    window = max(window, 30) if method == "advanced" else max(window, 10)
    if len(df) < window * 2:
        print(f"[Warning] Not enough data for support/resistance detection. Need at least {window*2} candles, got {len(df)}")
        return {"support": [], "resistance": []}

    for i in range(window, len(df) - window):
        local_lows = df['Low'].iloc[i - window:i + window]
        if df['Low'].iloc[i] == local_lows.min():
            level = df['Low'].iloc[i]
            if not any(abs(level - s) / s < tolerance for s in supports):
                supports.append(level)
        local_highs = df['High'].iloc[i - window:i + window]
        if df['High'].iloc[i] == local_highs.max():
            level = df['High'].iloc[i]
            if not any(abs(level - r) / r < tolerance for r in resistances):
                resistances.append(level)

    supports = sorted(supports)
    resistances = sorted(resistances)
    return {
        "support": supports[-3:] if supports else [],
        "resistance": resistances[-3:] if resistances else []
    }

# =========================
# IV Metrics Calculation
# =========================
def calculate_iv_metrics(ticker, data):
    """
    Calculate implied volatility metrics from historical data.
    """
    try:
        if 'volatility' not in data.columns:
            data['returns'] = data['Close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=21).std() * (252 ** 0.5)
        current_iv = data['volatility'].iloc[-1] * 100
        iv_series = data['volatility'].dropna()
        iv_rank = (iv_series.rank(pct=True).iloc[-1]) * 100
        return {
            'iv_rank': iv_rank,
            'iv_percentile': iv_rank,
            'hv_30': current_iv,
            'vix': 20.0  # Placeholder
        }
    except Exception as e:
        print(f"[Warning] Could not calculate IV metrics: {e}")
        return {
            'iv_rank': 0,
            'iv_percentile': 0,
            'hv_30': 0,
            'vix': 20.0
        }

# =========================
# Options Flow (Placeholder)
# =========================
def get_options_flow(ticker):
    """
    Get options flow data (placeholder).
    """
    print(f"[Info] Options flow data not available for {ticker}")
    return None
