# =========================
# Imports and Setup
# =========================
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import inspect
from typing import Optional, List, Dict, Any, Callable, Union

# Optional pandas_ta import (not required for current implementation)
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logging.warning("pandas_ta not available - using custom indicator calculations")

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

def get_supported_timeframes(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
            return {}
            
        # Check required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in data.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return {}
            
        # Remove rows with NaN values
        clean_data = data.dropna(subset=required)
        if clean_data.empty:
            logger.error("No valid data after dropping NaN values")
            return {}
            
        try:
            # Calculate base metrics for each timeframe
            result = {}
            for tf_key, tf_value in timeframes.items():
                result[tf_key] = calculate_indicators(data, timeframe=tf_key)
            return result
            
        except Exception as e:
            logger.error(f"Error calculating timeframe metrics: {str(e)}")
            return {}
            
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        return {}

def log_calculated_indicators(df: pd.DataFrame, timeframe: str):
    """
    Log all calculated technical indicators with their latest values
    
    Args:
        df (pd.DataFrame): DataFrame with calculated indicators
        timeframe (str): Current timeframe being analyzed
    """
    try:
        if df.empty:
            logger.warning("No data available for indicator logging")
            return
            
        # Get the latest row for current values
        latest = df.iloc[-1]
        
        # Categorize indicators
        indicator_categories = {
            '📈 Trend Indicators': {
                'SMA_20': 'SMA(20)',
                'SMA_50': 'SMA(50)', 
                'EMA_20': 'EMA(20)',
                'EMA_50': 'EMA(50)',
                'VWAP': 'VWAP',
                'ADX': 'ADX(14)'
            },
            '⚡ Momentum Indicators': {
                'RSI': 'RSI(14)',
                'MACD': 'MACD Line',
                'MACD_Signal': 'MACD Signal'
            },
            '📊 Volatility Indicators': {
                'ATR': 'ATR(14)',
                'BB_upper': 'BB Upper',
                'BB_middle': 'BB Middle', 
                'BB_lower': 'BB Lower',
                'volatility': 'Historical Vol'
            },
            '💰 Volume Indicators': {
                'OBV': 'On-Balance Volume'
            },
            '📉 Market Data': {
                'returns': 'Daily Returns'
            }
        }
        
        # Log header
        logger.info(f"🔧 Technical Indicators Calculated ({timeframe} timeframe):")
        
        # Log each category
        for category, indicators in indicator_categories.items():
            calculated_indicators = []
            
            for col_name, display_name in indicators.items():
                if col_name in df.columns:
                    value = latest[col_name]
                    if pd.notna(value):
                        # Format value based on indicator type
                        if col_name in ['RSI', 'ADX']:
                            calculated_indicators.append(f"{display_name}: {value:.1f}")
                        elif col_name in ['returns', 'volatility']:
                            calculated_indicators.append(f"{display_name}: {value*100:.2f}%")
                        elif col_name == 'OBV':
                            calculated_indicators.append(f"{display_name}: {value:,.0f}")
                        else:
                            calculated_indicators.append(f"{display_name}: ${value:.2f}")
                    else:
                        calculated_indicators.append(f"{display_name}: N/A")
            
            if calculated_indicators:
                logger.info(f"   {category}: {' | '.join(calculated_indicators)}")
        
        # Log timeframe-specific indicators if they exist
        timeframe_indicators = []
        timeframe_cols = [f'RSI_{timeframe}', f'MACD_{timeframe}', f'ADX_{timeframe}', 
                         f'ATR_{timeframe}', f'OBV_{timeframe}']
        
        for col in timeframe_cols:
            if col in df.columns and pd.notna(latest[col]):
                base_name = col.replace(f'_{timeframe}', '')
                value = latest[col]
                if base_name in ['RSI', 'ADX']:
                    timeframe_indicators.append(f"{base_name}: {value:.1f}")
                elif base_name == 'OBV':
                    timeframe_indicators.append(f"{base_name}: {value:,.0f}")
                else:
                    timeframe_indicators.append(f"{base_name}: ${value:.2f}")
        
        if timeframe_indicators:
            logger.info(f"   🕐 {timeframe.upper()} Specific: {' | '.join(timeframe_indicators)}")
        
        # Summary statistics
        total_indicators = len([col for col in df.columns 
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']])
        
        valid_indicators = len([col for col in df.columns 
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'] 
                              and pd.notna(latest.get(col))])
        
        logger.info(f"✅ Summary: {valid_indicators}/{total_indicators} indicators calculated successfully")
        
    except Exception as e:
        logger.error(f"Error logging indicators: {str(e)}")

def calculate_indicators(data: pd.DataFrame, 
                       timeframe: str = "1d",
                       strategy_type: Optional[str] = None,
                       selected_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate technical indicators for market analysis.
    
    Calculates indicators to match yfinance column structure:
    ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
     'returns', 'volatility', 'RSI', 'RSI_1d', 'MACD', 'MACD_Signal', 
     'MACD_1d', 'MACD_Signal_1d', 'ADX', 'ADX_1d', 'OBV', 'OBV_1d', 
     'ATR', 'ATR_1d', 'EMA_20', 'EMA_50', 'SMA_20', 'SMA_50', 
     'BB_upper', 'BB_middle', 'BB_lower', 'VWAP']
    
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
        df = data.copy()
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            logger.error("Missing required OHLCV columns")
            return pd.DataFrame()
            
        if not selected_indicators:
            selected_indicators = ["RSI", "MACD", "BB", "ADX", "ATR", "OBV"]

        # --- Returns & Volatility ---
        df['returns'] = df['Close'].pct_change()
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['volatility'] = (df['returns']
                            .rolling(window=21, min_periods=21)
                            .std() * np.sqrt(252))

        # --- Moving Averages ---
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False, min_periods=50).mean()

        # --- VWAP ---
        if 'Volume' in df.columns and 'Close' in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # --- Bollinger Bands ---
        rolling_mean = df['Close'].rolling(window=20, min_periods=20).mean()
        rolling_std = df['Close'].rolling(window=20, min_periods=20).std()
        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_middle'] = rolling_mean
        df['BB_lower'] = rolling_mean - (rolling_std * 2)

        # --- RSI (14) ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add timeframe-specific RSI
        df[f'RSI_{timeframe}'] = df['RSI']

        # --- ATR (14) ---
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14, min_periods=14).mean()
        
        # Add timeframe-specific ATR
        df[f'ATR_{timeframe}'] = df['ATR']

        # --- OBV ---
        df['OBV'] = (df['Volume'] * ((df['Close'] > df['Close'].shift(1)).astype(int) - 
                                   (df['Close'] < df['Close'].shift(1)).astype(int))).cumsum()
        
        # Add timeframe-specific OBV
        df[f'OBV_{timeframe}'] = df['OBV']

        # --- MACD ---
        ema12 = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
        
        # Add timeframe-specific MACD
        df[f'MACD_{timeframe}'] = df['MACD']
        df[f'MACD_Signal_{timeframe}'] = df['MACD_Signal']

        # --- ADX (14) ---
        try:
            # Calculate +DI and -DI
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_dm_series = pd.Series(plus_dm, index=df.index)
            minus_dm_series = pd.Series(minus_dm, index=df.index)
            
            tr = true_range  # Already calculated above
            
            # 14-period smoothed averages
            plus_di = 100 * (plus_dm_series.rolling(window=14, min_periods=14).mean() / 
                            tr.rolling(window=14, min_periods=14).mean())
            minus_di = 100 * (minus_dm_series.rolling(window=14, min_periods=14).mean() / 
                             tr.rolling(window=14, min_periods=14).mean())
            
            # Calculate DX
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
            
            # Calculate ADX as 14-period smoothed average of DX
            df['ADX'] = dx.rolling(window=14, min_periods=14).mean()
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            df['ADX'] = np.nan
            
        # Add timeframe-specific ADX
        df[f'ADX_{timeframe}'] = df['ADX']

        # Ensure all required columns exist for downstream code
        required_cols = [
            'EMA_20', 'EMA_50', 'OBV', 'ATR', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ADX',
            'BB_upper', 'BB_middle', 'BB_lower', 'VWAP', 'Volume', 'returns', 'volatility',
            f'RSI_{timeframe}', f'MACD_{timeframe}', f'MACD_Signal_{timeframe}', 
            f'ADX_{timeframe}', f'OBV_{timeframe}', f'ATR_{timeframe}'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                if col == 'Volume' and 'Volume' in data.columns:
                    df[col] = data['Volume']
                else:
                    df[col] = np.nan

        # Log all calculated indicators with their latest values
        log_calculated_indicators(df, timeframe)
        
        return df

    except Exception as e:
        logger.error(f"Error in indicator calculation setup: {str(e)}")
        return pd.DataFrame()

# =========================
# Support & Resistance Detection
# =========================
def detect_support_resistance(data: pd.DataFrame, method: str = "quick", 
                             window: int = 20, tolerance: float = 0.01) -> Dict[str, List[float]]:
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
        logger.warning(f"Not enough data for support/resistance detection. Need at least {window*2} candles, got {len(df)}")
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
def calculate_iv_metrics(ticker: str, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate implied volatility metrics from historical data.
    
    Args:
        ticker (str): Stock ticker symbol
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        dict: Dictionary containing IV metrics
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
            'vix': 20.0  # Placeholder - would need VIX data from external source
        }
    except Exception as e:
        logger.warning(f"Could not calculate IV metrics for {ticker}: {e}")
        return {
            'iv_rank': 0,
            'iv_percentile': 0,
            'hv_30': 0,
            'vix': 20.0
        }

# =========================
# Options Flow (Placeholder)
# =========================
def get_options_flow(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get options flow data (placeholder).
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Optional[Dict[str, Any]]: Options flow data or None
    """
    logging.debug(f"⚠️ Options flow data not implemented for {ticker}")  # Changed to debug level
    return None
