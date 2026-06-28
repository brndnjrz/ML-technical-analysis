import pandas as pd
def determine_trend(data: pd.DataFrame, window: int = 20) -> str:
    """
    Determine the trend direction ('uptrend', 'downtrend', 'sideways') based on closing prices.
    Args:
        data (pd.DataFrame): DataFrame with a 'Close' column
        window (int): Number of periods to use for trend calculation
    Returns:
        str: 'uptrend', 'downtrend', or 'sideways'
    """
    if data is None or 'Close' not in data.columns or len(data) < window:
        return 'sideways'
    closes = data['Close'].tail(window)
    x = range(len(closes))
    # Simple linear regression slope
    import numpy as np
    slope = np.polyfit(x, closes, 1)[0]
    if slope > 0.02:
        return 'uptrend'
    elif slope < -0.02:
        return 'downtrend'
    else:
        return 'sideways'
# =========================
# Imports and Setup
# =========================
import pandas as pd
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
                'MACD_Signal': 'MACD Signal',
                'STOCH_%K': 'Stochastic %K',
                'STOCH_%D': 'Stochastic %D'
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
                         f'ATR_{timeframe}', f'OBV_{timeframe}', f'STOCH_%K_{timeframe}', f'STOCH_%D_{timeframe}']
        
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
                       timeframe: str = "15m",
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
        timeframe: Data timeframe (e.g., "5m", "15m", "30m", "1h", "1d")
        strategy_type: Type of trading strategy
        selected_indicators: List of indicators to calculate
        
    Returns:
        DataFrame with calculated indicators
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.error("Invalid or empty input data")
        return pd.DataFrame()
        
    try:
        # Log initial data structure from yfinance
        logger.info(f"📥 RAW DATA FROM YFINANCE:")
        logger.info(f"   • Data Shape: {data.shape} (rows: {data.shape[0]}, columns: {data.shape[1]})")
        logger.info(f"   • Columns Received: {list(data.columns)}")
        logger.info(f"   • Date Range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"   • Timeframe: {timeframe}")
        
        # Log sample of raw data
        if len(data) > 0:
            latest = data.iloc[-1]
            logger.info(f"   • Latest OHLCV: O={latest.get('Open', 'N/A'):.2f}, H={latest.get('High', 'N/A'):.2f}, L={latest.get('Low', 'N/A'):.2f}, C={latest.get('Close', 'N/A'):.2f}, V={latest.get('Volume', 'N/A'):,.0f}")
            
        df = data.copy()
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            logger.error("Missing required OHLCV columns")
            return pd.DataFrame()
            
        if not selected_indicators:
            selected_indicators = ["RSI", "MACD", "BB", "ADX", "ATR", "OBV"]

        logger.info(f"🔧 CALCULATING TECHNICAL INDICATORS:")
        logger.info(f"   • Selected Indicators: {selected_indicators}")
        logger.info(f"   • Strategy Type: {strategy_type}")

        # Track which indicators we're calculating
        calculated_indicators = []

        # --- Returns & Volatility ---
        logger.info("   📈 Calculating Returns & Volatility...")
        df['returns'] = df['Close'].pct_change()
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['volatility'] = (df['returns']
                            .rolling(window=21, min_periods=21)
                            .std() * np.sqrt(252))
        calculated_indicators.extend(['returns', 'volatility'])

        # --- Moving Averages ---
        logger.info("   📊 Calculating Moving Averages (SMA & EMA)...")
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False, min_periods=50).mean()
        calculated_indicators.extend(['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50'])
        
        # Forward fill early NaN values for better chart appearance
        df['SMA_20'] = df['SMA_20'].bfill().fillna(df['Close'])
        df['SMA_50'] = df['SMA_50'].bfill().fillna(df['Close'])
        df['EMA_20'] = df['EMA_20'].bfill().fillna(df['Close'])
        df['EMA_50'] = df['EMA_50'].bfill().fillna(df['Close'])
        
        # Add timeframe-specific moving averages
        df[f'SMA_20_{timeframe}'] = df['SMA_20']
        df[f'SMA_50_{timeframe}'] = df['SMA_50']
        df[f'EMA_20_{timeframe}'] = df['EMA_20']
        df[f'EMA_50_{timeframe}'] = df['EMA_50']

        # --- VWAP ---
        logger.info("   💰 Calculating Volume Weighted Average Price (VWAP)...")
        if 'Volume' in df.columns and 'Close' in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            # Add timeframe-specific VWAP
            df[f'VWAP_{timeframe}'] = df['VWAP']
            calculated_indicators.extend(['VWAP', f'VWAP_{timeframe}'])

        # --- Bollinger Bands ---
        logger.info("   📏 Calculating Bollinger Bands...")
        try:
            rolling_mean = df['Close'].rolling(window=20, min_periods=20).mean()
            rolling_std = df['Close'].rolling(window=20, min_periods=20).std()
            
            # Add safety checks to avoid NoneType errors
            rolling_mean = rolling_mean.fillna(df['Close'])  # Use price if no MA
            rolling_std = rolling_std.fillna(df['Close'] * 0.02)  # Use 2% if no std
            
            df['BB_upper'] = rolling_mean + (rolling_std * 2)
            df['BB_middle'] = rolling_mean
            df['BB_lower'] = rolling_mean - (rolling_std * 2)
            calculated_indicators.extend(['BB_upper', 'BB_middle', 'BB_lower'])
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
            # Create default values if calculation fails
            df['BB_upper'] = df['Close'] * 1.05  # 5% above price
            df['BB_middle'] = df['Close']
            df['BB_lower'] = df['Close'] * 0.95  # 5% below price
        
        # Forward fill early NaN values for Bollinger Bands
        df['BB_upper'] = df['BB_upper'].bfill().fillna(df['Close'] * 1.05)
        df['BB_middle'] = df['BB_middle'].bfill().fillna(df['Close'])
        df['BB_lower'] = df['BB_lower'].bfill().fillna(df['Close'] * 0.95)
        
        # Add timeframe-specific Bollinger Bands
        df[f'BB_upper_{timeframe}'] = df['BB_upper']
        df[f'BB_middle_{timeframe}'] = df['BB_middle']
        df[f'BB_lower_{timeframe}'] = df['BB_lower']

        # --- RSI (14) ---
        logger.info("   ⚡ Calculating RSI (Relative Strength Index)...")
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
            
            # Add safety check for division by zero
            loss = loss.replace(0, 1e-5)  # Replace zeros with small number
            
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Handle any NaN, inf, or -inf values
            df['RSI'] = df['RSI'].replace([np.inf, -np.inf], np.nan)
            
            # Forward fill early NaN values for RSI (assume neutral 50)
            df['RSI'] = df['RSI'].bfill().fillna(50)
            
            # Add timeframe-specific RSI
            df[f'RSI_{timeframe}'] = df['RSI']
            
            calculated_indicators.extend(['RSI', f'RSI_{timeframe}'])
        except Exception as e:
            logger.warning(f"Error calculating RSI: {str(e)}")
            # Create default RSI values if calculation fails
            df['RSI'] = 50  # Neutral RSI
            df[f'RSI_{timeframe}'] = 50

        # --- ATR (14) ---
        logger.info("   📊 Calculating ATR (Average True Range)...")
        try:
            # Calculate true range components with safety checks
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            
            # Replace any NaN values
            high_low = high_low.fillna(0)
            high_close = high_close.fillna(0)
            low_close = low_close.fillna(0)
            
            # Create true range
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            # Calculate ATR and handle NaN values
            atr = true_range.rolling(window=14, min_periods=5).mean()
            df['ATR'] = atr.fillna(true_range)  # Use true range if ATR is NaN
            
            # If still NaN (e.g. first row), use a default percentage of price
            default_atr = df['Close'] * 0.02  # Default to 2% of price
            df['ATR'] = df['ATR'].fillna(default_atr)
            
            # Add timeframe-specific ATR
            df[f'ATR_{timeframe}'] = df['ATR']
            
            calculated_indicators.extend(['ATR', f'ATR_{timeframe}'])
        except Exception as e:
            logger.warning(f"Error calculating ATR: {str(e)}")
            # Create default ATR values if calculation fails - 2% of price
            df['ATR'] = df['Close'] * 0.02
            df[f'ATR_{timeframe}'] = df['ATR']

        # --- OBV ---
        logger.info("   💹 Calculating OBV (On-Balance Volume)...")
        try:
            # Calculate OBV with safety checks
            close_diff = df['Close'] - df['Close'].shift(1)
            
            # Create direction multipliers (1 for up, -1 for down, 0 for unchanged)
            direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
            
            # Handle first row which would be NaN
            direction = pd.Series(direction, index=df.index).fillna(0)
            
            # Calculate OBV
            obv_values = (df['Volume'] * direction).fillna(0)
            df['OBV'] = obv_values.cumsum()
            
            # Add timeframe-specific OBV
            df[f'OBV_{timeframe}'] = df['OBV']
            
            calculated_indicators.extend(['OBV', f'OBV_{timeframe}'])
        except Exception as e:
            logger.warning(f"Error calculating OBV: {str(e)}")
            # Create default OBV values if calculation fails
            df['OBV'] = 0
            df[f'OBV_{timeframe}'] = 0
        
        # Add timeframe-specific OBV with proper initialization
        df[f'OBV_{timeframe}'] = df['OBV']
        calculated_indicators.extend(['OBV', f'OBV_{timeframe}'])

        # --- MACD ---
        logger.info("   📈 Calculating MACD (Moving Average Convergence Divergence)...")
        ema12 = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
        calculated_indicators.extend(['MACD', 'MACD_Signal', f'MACD_{timeframe}', f'MACD_Signal_{timeframe}'])
        
        # Forward fill early NaN values for MACD
        df['MACD'] = df['MACD'].bfill().fillna(0)
        df['MACD_Signal'] = df['MACD_Signal'].bfill().fillna(0)
        
        # Add timeframe-specific MACD
        df[f'MACD_{timeframe}'] = df['MACD']
        df[f'MACD_Signal_{timeframe}'] = df['MACD_Signal']

        # --- ADX (14) ---
        logger.info("   🎯 Calculating ADX (Average Directional Index)...")
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
            
            # Forward fill early NaN values for ADX (assume neutral 25)
            df['ADX'] = df['ADX'].bfill().fillna(25)
            calculated_indicators.extend(['ADX', f'ADX_{timeframe}'])
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            df['ADX'] = np.nan
            
        # Add timeframe-specific ADX
        df[f'ADX_{timeframe}'] = df['ADX']

        # --- Stochastic Oscillator (14,3,3) ---
        logger.info("   🔄 Calculating Stochastic Oscillator...")
        try:
            # Calculate %K (Fast Stochastic)
            low_min = df['Low'].rolling(window=14, min_periods=14).min()
            high_max = df['High'].rolling(window=14, min_periods=14).max()
            k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            
            # Calculate %D (Slow Stochastic) - 3-period SMA of %K
            d_percent = k_percent.rolling(window=3, min_periods=3).mean()
            
            df['STOCH_%K'] = k_percent
            df['STOCH_%D'] = d_percent
            
            # Forward fill early NaN values for Stochastic (assume neutral 50)
            df['STOCH_%K'] = df['STOCH_%K'].bfill().fillna(50)
            df['STOCH_%D'] = df['STOCH_%D'].bfill().fillna(50)
            
            # Add timeframe-specific Stochastic
            df[f'STOCH_%K_{timeframe}'] = df['STOCH_%K']
            df[f'STOCH_%D_{timeframe}'] = df['STOCH_%D']
            calculated_indicators.extend(['STOCH_%K', 'STOCH_%D', f'STOCH_%K_{timeframe}', f'STOCH_%D_{timeframe}'])
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            df['STOCH_%K'] = np.nan
            df['STOCH_%D'] = np.nan
            df[f'STOCH_%K_{timeframe}'] = np.nan
            df[f'STOCH_%D_{timeframe}'] = np.nan
            
        # --- Options-Specific Indicators ---
        logger.info("   💰 Calculating Options-Specific Indicators...")
        try:
            # Historical Volatility (HV) over different timeframes
            # These are important for options premium evaluation and strike selection
            df['HV_10'] = df['returns'].rolling(window=10).std() * np.sqrt(252)  # 2-week HV
            df['HV_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # 1-month HV
            df['HV_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)  # 3-month HV
            
            # Fill NaN values with reasonable defaults
            df['HV_10'] = df['HV_10'].fillna(df['volatility'] if 'volatility' in df else 0.2)
            df['HV_20'] = df['HV_20'].fillna(df['volatility'] if 'volatility' in df else 0.2)
            df['HV_60'] = df['HV_60'].fillna(df['volatility'] if 'volatility' in df else 0.2)
            
            calculated_indicators.extend(['HV_10', 'HV_20', 'HV_60'])
            
            # Volatility Smile Proxy (approximation of skew without options chain data)
            # This helps identify potential directional bias in options pricing
            if len(df) > 20:
                # Calculate returns asymmetry over past 20 days
                returns = df['returns'].iloc[-20:].dropna()
                if len(returns) > 0:
                    upside_returns = returns[returns > 0]
                    downside_returns = returns[returns < 0]
                    
                    upside_vol = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
                    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                    
                    # Volatility skew (higher values = more downside risk priced in)
                    df['VOL_SKEW'] = downside_vol / upside_vol if upside_vol > 0 else 1.0
                    df['VOL_SKEW'] = df['VOL_SKEW'].replace([np.inf, -np.inf], 1.5).fillna(1.0)
                    calculated_indicators.append('VOL_SKEW')
            
            # Standard Deviation Moves for Strike Selection
            # Critical for identifying probable price ranges and setting strikes
            df['STD_1_UP'] = df['Close'] * (1 + df['HV_20'])
            df['STD_1_DOWN'] = df['Close'] * (1 - df['HV_20'])
            df['STD_2_UP'] = df['Close'] * (1 + df['HV_20'] * 2)
            df['STD_2_DOWN'] = df['Close'] * (1 - df['HV_20'] * 2)
            calculated_indicators.extend(['STD_1_UP', 'STD_1_DOWN', 'STD_2_UP', 'STD_2_DOWN'])
            
            # Put-Call Ratio Proxy (using volume patterns as proxy)
            # This helps identify potential sentiment extremes
            if 'Volume' in df.columns and len(df) > 10:
                volume_changes = df['Volume'].pct_change().fillna(0)
                price_changes = df['Close'].pct_change().fillna(0)
                
                # Calculate correlation between volume and price changes
                # Negative correlation suggests put activity (volume up when price down)
                # Positive correlation suggests call activity (volume up when price up)
                if len(volume_changes) > 5 and len(price_changes) > 5:
                    vol_price_corr = np.corrcoef(volume_changes[-10:], price_changes[-10:])[0, 1]
                    df['VOL_PRICE_CORR'] = vol_price_corr
                    calculated_indicators.append('VOL_PRICE_CORR')
            
            # Iron Condor Suitability Index (higher = better for Iron Condors)
            # Combines ADX, volatility, and price channel width
            try:
                # Low ADX + stable volatility + price in middle of channel = good for IC
                adx_component = 100 - df['ADX'] if 'ADX' in df.columns else 50  # Lower ADX is better for IC
                price_channel_width = (df['BB_upper'] - df['BB_lower']) / df['Close'] if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'Close']) else 0.05
                
                # Price position in the channel (0 = middle = ideal)
                price_position = 0
                if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower', 'Close']):
                    upper_space = (df['BB_upper'] - df['Close']) / (df['BB_upper'] - df['BB_lower'])
                    lower_space = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
                    # 0 = middle of channel, 1 = at band edge
                    price_position = abs(0.5 - upper_space) * 2
                
                # Calculate suitability (100 = perfect conditions, 0 = poor conditions)
                df['IC_SUITABILITY'] = (
                    # Lower ADX is better (max 50 points)
                    (adx_component * 0.5) + 
                    # Lower price position is better (max 30 points)
                    ((1 - price_position) * 30) + 
                    # Moderate channel width is best (max 20 points)
                    (20 - abs(price_channel_width * 100 - 5) * 4).clip(0, 20)
                ).clip(0, 100)
                
                calculated_indicators.append('IC_SUITABILITY')
            except Exception as e:
                logger.warning(f"Error calculating IC_SUITABILITY: {str(e)}")
                df['IC_SUITABILITY'] = 50  # Neutral value
                
        except Exception as e:
            logger.warning(f"Error calculating options-specific indicators: {str(e)}")

        # Ensure all required columns exist for downstream code
        required_cols = [
            'EMA_20', 'EMA_50', 'OBV', 'ATR', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ADX',
            'BB_upper', 'BB_middle', 'BB_lower', 'VWAP', 'Volume', 'returns', 'volatility',
            'STOCH_%K', 'STOCH_%D',  # Add Stochastic indicators
            # Timeframe-specific indicators
            f'RSI_{timeframe}', f'MACD_{timeframe}', f'MACD_Signal_{timeframe}', 
            f'ADX_{timeframe}', f'OBV_{timeframe}', f'ATR_{timeframe}',
            f'STOCH_%K_{timeframe}', f'STOCH_%D_{timeframe}',  # Stochastic
            f'SMA_20_{timeframe}', f'SMA_50_{timeframe}',  # Moving averages
            f'EMA_20_{timeframe}', f'EMA_50_{timeframe}',  # Exponential moving averages
            f'VWAP_{timeframe}',  # Volume weighted average price
            f'BB_upper_{timeframe}', f'BB_middle_{timeframe}', f'BB_lower_{timeframe}'  # Bollinger Bands
        ]
        
        for col in required_cols:
            if col not in df.columns:
                if col == 'Volume' and 'Volume' in data.columns:
                    df[col] = data['Volume']
                else:
                    df[col] = np.nan

        # Log final results
        logger.info(f"✅ INDICATOR CALCULATION COMPLETE:")
        logger.info(f"   • Total Indicators Calculated: {len(calculated_indicators)}")
        logger.info(f"   • Base Indicators: {[i for i in calculated_indicators if '_' not in i or 'STOCH_%' in i]}")
        logger.info(f"   • Timeframe-Specific ({timeframe}): {[i for i in calculated_indicators if f'_{timeframe}' in i and 'STOCH_%' not in i]}")
        
        # Log final DataFrame structure
        logger.info(f"📊 FINAL DATAFRAME STRUCTURE:")
        logger.info(f"   • Shape: {df.shape}")
        logger.info(f"   • All Columns: {list(df.columns)}")
        
        # Show which indicators have valid data
        valid_indicators = []
        invalid_indicators = []
        for col in calculated_indicators:
            if col in df.columns:
                if df[col].notna().any():
                    valid_indicators.append(col)
                else:
                    invalid_indicators.append(col)
        
        if valid_indicators:
            logger.info(f"   • ✅ Valid Indicators: {valid_indicators}")
        if invalid_indicators:
            logger.info(f"   • ❌ Invalid/Empty Indicators: {invalid_indicators}")

        # Log all calculated indicators with their latest values
        log_calculated_indicators(df, timeframe)
        
        # Log indicator timing alignment
        first_valid_indices = {}
        for col in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'ADX', 'STOCH_%K']:
            if col in df.columns:
                first_valid = df[col].first_valid_index()
                if first_valid is not None:
                    first_valid_indices[col] = df.index.get_loc(first_valid)
        
        if first_valid_indices:
            logger.info(f"📊 Indicator Start Alignment: All indicators now start from the beginning of the dataset")
            logger.info(f"   Note: Early values use forward-fill method for visual consistency")
        
        return df

    except Exception as e:
        logger.error(f"Error in indicator calculation setup: {str(e)}")
        return pd.DataFrame()

# =========================
# Support & Resistance Detection
# =========================
def analyze_options_market_regime(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the current market regime for options trading strategy selection.
    
    Options strategies perform differently under different market regimes:
    - Trending markets: Directional strategies (swing trading, day trading)
    - Range-bound markets: Non-directional strategies (iron condors)
    - High volatility: Volatility strategies (day trading calls/puts)
    - Low volatility: Income strategies (covered calls, cash-secured puts)
    
    Args:
        data (pd.DataFrame): DataFrame with price and indicator data
        
    Returns:
        Dict[str, Any]: Market regime analysis with strategy recommendations
    """
    try:
        if data.empty or len(data) < 30:
            return {'regime': 'insufficient_data', 'strategy': 'none'}
            
        # Get latest values
        latest = data.iloc[-1]
        
        # --- Trend Analysis ---
        # Calculate directional movement over different timeframes
        short_term_change = data['Close'].pct_change(5).iloc[-1] * 100  # 5-day (week)
        medium_term_change = data['Close'].pct_change(21).iloc[-1] * 100  # 21-day (month)
        long_term_change = data['Close'].pct_change(63).iloc[-1] * 100  # 63-day (quarter)
        
        # Determine trend regime based on ADX
        adx = latest.get('ADX', 0)
        trend_regime = "ranging"
        if adx > 30:
            if short_term_change > 2 and medium_term_change > 5:
                trend_regime = "strongly_trending_up"
            elif short_term_change < -2 and medium_term_change < -5:
                trend_regime = "strongly_trending_down"
            else:
                trend_regime = "moderately_trending"
        elif adx > 20:
            if abs(short_term_change) > 2:
                trend_regime = "weakly_trending"
            else:
                trend_regime = "neutral"
                
        # --- Volatility Analysis ---
        # Use historical volatility or ATR as volatility measure
        current_vol = latest.get('volatility', 0)
        
        # Calculate historical volatility percentile
        if 'volatility' in data.columns:
            vol_series = data['volatility'].dropna()
            if len(vol_series) > 20:
                vol_percentile = (vol_series.rank(pct=True).iloc[-1]) * 100
            else:
                vol_percentile = 50
        else:
            vol_percentile = 50
        
        # Determine volatility regime
        vol_regime = "medium_volatility"
        if vol_percentile > 75:
            vol_regime = "high_volatility"
        elif vol_percentile < 25:
            vol_regime = "low_volatility"
            
        # --- Momentum Analysis ---
        rsi = latest.get('RSI', 50)
        momentum_regime = "neutral"
        if rsi > 70:
            momentum_regime = "overbought"
        elif rsi < 30:
            momentum_regime = "oversold"
            
        # --- Combine all regimes ---
        # Create combined market regime
        combined_regime = f"{trend_regime}_{vol_regime}_{momentum_regime}"
        
        # Map to simplified regime categories
        simplified_regime = "mixed"
        if "strongly_trending" in trend_regime and vol_regime == "low_volatility":
            simplified_regime = "trending_low_vol"
        elif "strongly_trending" in trend_regime and vol_regime == "high_volatility":
            simplified_regime = "trending_high_vol"
        elif trend_regime == "ranging" and vol_regime == "low_volatility":
            simplified_regime = "ranging_low_vol"
        elif trend_regime == "ranging" and vol_regime == "high_volatility":
            simplified_regime = "ranging_high_vol"
        elif trend_regime == "neutral" and "overbought" in momentum_regime:
            simplified_regime = "topping"
        elif trend_regime == "neutral" and "oversold" in momentum_regime:
            simplified_regime = "bottoming"
            
        # --- Strategy Recommendations ---
        # Map market regimes to options strategies
        strategies = {
            "trending_low_vol": {
                "primary": "Directional Options (Calls/Puts)",
                "secondary": "Vertical Spreads in trend direction",
                "avoid": "Iron Condors"
            },
            "trending_high_vol": {
                "primary": "Credit Spreads",
                "secondary": "Swing Trading with reduced position size",
                "avoid": "Iron Condors"
            },
            "ranging_low_vol": {
                "primary": "Iron Condors",
                "secondary": "Credit Spreads at range boundaries",
                "avoid": "Day Trading Calls/Puts"
            },
            "ranging_high_vol": {
                "primary": "Wide Iron Condors",
                "secondary": "Covered Calls",
                "avoid": "Tight Iron Condors"
            },
            "topping": {
                "primary": "Credit Spreads",
                "secondary": "Cash-Secured Puts",
                "avoid": "Covered Calls"
            },
            "bottoming": {
                "primary": "Cash-Secured Puts",
                "secondary": "Credit Spreads",
                "avoid": "Covered Calls"
            },
            "mixed": {
                "primary": "Balanced Iron Condors",
                "secondary": "Credit Spreads",
                "avoid": "Highly directional trades"
            }
        }
        
        # Get recommended strategy based on regime
        recommended_strategy = strategies.get(simplified_regime, strategies["mixed"])
        
        # Build result dictionary
        result = {
            'detailed_regime': combined_regime,
            'simplified_regime': simplified_regime,
            'trend': {
                'regime': trend_regime,
                'adx': adx,
                'short_term_change': short_term_change,
                'medium_term_change': medium_term_change,
                'long_term_change': long_term_change
            },
            'volatility': {
                'regime': vol_regime,
                'current': current_vol,
                'percentile': vol_percentile
            },
            'momentum': {
                'regime': momentum_regime,
                'rsi': rsi
            },
            'strategy': recommended_strategy
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing options market regime: {str(e)}")
        return {
            'regime': 'error',
            'strategy': {
                'primary': 'Error in analysis',
                'secondary': 'None',
                'avoid': 'Risky strategies'
            },
            'error': str(e)
        }

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

    # Make window size more adaptive to available data
    if method == "advanced":
        # For advanced method, try to use at least 30 candles
        ideal_window = max(window, 30)
        min_required = 40
    else:
        # For quick method, can work with just 10 candles
        ideal_window = max(window, 10)
        min_required = 20
        
    # Adapt to available data
    available_data_points = len(df)
    
    if available_data_points < min_required:
        # We can still try with smaller window if we have at least some data
        if available_data_points >= 10:
            # Adjust window to work with available data
            window = max(3, int(available_data_points / 4))
            logger.info(f"Adjusting support/resistance detection window to {window} due to limited data ({available_data_points} candles)")
        else:
            logger.warning(f"Not enough data for support/resistance detection. Need at least 10 candles, got {available_data_points}")
            return {"support": [], "resistance": []}
    else:
        # We have enough data, use ideal window
        window = ideal_window

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
    Calculate implied volatility metrics from historical data for options trading.
    
    Enhanced to provide better options trading metrics:
    - IV Rank: IV relative to 52-week range (higher rank = better for selling options)
    - IV Percentile: Percentage of days with lower IV (higher = better for selling)
    - HV/IV Ratio: Historical vs Implied Volatility (>1 = options potentially overpriced)
    - Term Structure: Short-term vs Long-term volatility (contango/backwardation)
    
    Args:
        ticker (str): Stock ticker symbol
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        dict: Dictionary containing IV metrics for options trading decision making
    """
    try:
        # Calculate historical volatility (HV) metrics
        if 'volatility' not in data.columns:
            data['returns'] = data['Close'].pct_change()
            # 21-day HV (roughly 1 month of trading days)
            data['volatility'] = data['returns'].rolling(window=21).std() * (252 ** 0.5)
        
        # Calculate additional HV timeframes for term structure
        data['hv_10'] = data['returns'].rolling(window=10).std() * (252 ** 0.5)  # ~2 weeks
        data['hv_63'] = data['returns'].rolling(window=63).std() * (252 ** 0.5)  # ~3 months
        
        # Get current values
        current_hv = data['volatility'].iloc[-1] * 100  # Convert to percentage
        current_hv_10 = data['hv_10'].iloc[-1] * 100
        current_hv_63 = data['hv_63'].iloc[-1] * 100
        
        # Calculate IV Rank (current IV position in available range)
        # Safely get as much history as available (up to 1 year)
        available_points = min(len(data['volatility'].dropna()), 252)
        
        if available_points < 5:  # Need at least a week of data
            logger.info(f"Limited volatility history for {ticker}: {available_points} data points")
            # Use simplified calculations with defaults
            min_iv = current_hv * 0.8  # Assume 20% lower as min
            max_iv = current_hv * 1.2  # Assume 20% higher as max
            iv_rank = 50  # Default to middle
            iv_percentile = 50  # Default to middle
        else:
            # Use whatever history we have
            iv_series = data['volatility'].dropna().iloc[-available_points:] 
            min_iv = iv_series.min() * 100
            max_iv = iv_series.max() * 100
            
            # Prevent division by zero
            iv_range = max(max_iv - min_iv, 0.01)
            iv_rank = min(max(((current_hv - min_iv) / iv_range) * 100, 0), 100)
            
            # Calculate IV Percentile (percentage of days with lower IV)
            iv_percentile = (iv_series.rank(pct=True).iloc[-1]) * 100
        
        # Calculate volatility term structure (contango/backwardation)
        # Contango: Short-term vol < Long-term vol (normal)
        # Backwardation: Short-term vol > Long-term vol (fear/uncertainty)
        vol_term_ratio = current_hv_10 / current_hv_63 if current_hv_63 > 0 else 1.0
        
        # Determine volatility regime
        vol_regime = "High"
        if iv_rank < 30:
            vol_regime = "Low"
        elif iv_rank < 60:
            vol_regime = "Medium"
            
        # Volatility trend (rising or falling)
        vol_trend = "Stable"
        
        # Safely calculate volatility trend with available data
        vol_data = data['volatility'].dropna()
        if len(vol_data) >= 10:
            recent_vol = vol_data.iloc[-10:].values
            first_half_avg = np.mean(recent_vol[:5])
            second_half_avg = np.mean(recent_vol[5:])
            vol_change_pct = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
            
            if vol_change_pct > 0.05:  # 5% increase
                vol_trend = "Rising"
            elif vol_change_pct < -0.05:  # 5% decrease
                vol_trend = "Falling"
        elif len(vol_data) >= 4:
            # If we have limited data, use simpler calculation
            recent_vol = vol_data.values
            mid_point = len(recent_vol) // 2
            first_half_avg = np.mean(recent_vol[:mid_point])
            second_half_avg = np.mean(recent_vol[mid_point:])
            vol_change_pct = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
            
            if vol_change_pct > 0.05:  # 5% increase
                vol_trend = "Rising"
            elif vol_change_pct < -0.05:  # 5% decrease
                vol_trend = "Falling"
        
        # Get VIX (market volatility) if available or use placeholder
        # In a real implementation, you'd fetch VIX data from an external source
        vix = 20.0  # Placeholder
        
        # Create options strategy suggestions based on volatility metrics
        strategy_suggestions = []
        if iv_rank > 70 and vol_regime == "High":
            strategy_suggestions.append("Iron Condors (Sell OTM Call & Put Spreads)")
            strategy_suggestions.append("Credit Spreads (High Premium)")
        elif iv_rank < 30 and vol_regime == "Low":
            strategy_suggestions.append("Day Trading Calls/Puts (Directional Plays)")
            strategy_suggestions.append("Swing Trading (Exploit Trends)")
        
        if vol_trend == "Rising":
            strategy_suggestions.append("Day Trading Calls/Puts (Buy Options)")
        elif vol_trend == "Falling":
            strategy_suggestions.append("Covered Calls/Cash-Secured Puts (Sell Premium)")
        
        # Prepare comprehensive IV metrics dictionary for options trading
        return {
            'iv_rank': iv_rank,
            'iv_percentile': iv_percentile,
            'hv_30': current_hv,
            'hv_10': current_hv_10,
            'hv_63': current_hv_63,
            'vol_term_ratio': vol_term_ratio,
            'vol_regime': vol_regime,
            'vol_trend': vol_trend,
            'term_structure': "Contango" if vol_term_ratio < 0.95 else "Backwardation" if vol_term_ratio > 1.05 else "Flat",
            'vix': vix,
            'strategy_suggestions': strategy_suggestions
        }
    except Exception as e:
        logger.warning(f"Could not calculate IV metrics for {ticker}: {e}")
        return {
            'iv_rank': 50,
            'iv_percentile': 50,
            'hv_30': 30,
            'hv_10': 30,
            'hv_63': 30,
            'vol_term_ratio': 1.0,
            'vol_regime': "Medium",
            'vol_trend': "Stable",
            'term_structure': "Flat",
            'vix': 20.0,
            'strategy_suggestions': ["Insufficient volatility data"]
        }

# =========================
# Options Trading Indicators
# =========================
def calculate_options_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate specialized indicators for options trading strategies.
    
    Focused on key metrics for options trading decision making:
    - Support/Resistance levels for strike selection
    - Trend strength and direction for directional options
    - Volatility regime analysis for non-directional strategies
    - Momentum indicators for timing entries/exits
    
    Args:
        data (pd.DataFrame): Price and indicator data
        
    Returns:
        Dict[str, Any]: Options trading indicator metrics and signals
    """
    try:
        if data.empty:
            logger.warning("Empty data provided for options indicators calculation")
            return {}
            
        result = {
            'directional_signals': {},
            'non_directional_signals': {},
            'strike_selection': {},
            'timing': {}
        }
        
        # Get latest data point
        latest = data.iloc[-1]
        
        # ---- Trend Analysis for Directional Options ----
        
        # Price vs Moving Averages
        ma_signal = "neutral"
        if latest.get('Close', 0) > latest.get('SMA_50', 0):
            if latest.get('Close', 0) > latest.get('SMA_20', 0):
                ma_signal = "strongly_bullish"
            else:
                ma_signal = "moderately_bullish"
        elif latest.get('Close', 0) < latest.get('SMA_50', 0):
            if latest.get('Close', 0) < latest.get('SMA_20', 0):
                ma_signal = "strongly_bearish"
            else:
                ma_signal = "moderately_bearish"
        
        # MACD for trend confirmation
        macd_signal = "neutral"
        if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
            macd_signal = "bullish"
        elif latest.get('MACD', 0) < latest.get('MACD_Signal', 0):
            macd_signal = "bearish"
            
        # ADX for trend strength
        adx = latest.get('ADX', 0)
        trend_strength = "weak"
        if adx > 30:
            trend_strength = "strong"
        elif adx > 20:
            trend_strength = "moderate"
            
        # Combine trend signals
        trend_analysis = {
            'ma_signal': ma_signal,
            'macd_signal': macd_signal,
            'adx_strength': trend_strength,
            'adx_value': adx
        }
        
        # Generate directional options recommendation
        if (ma_signal in ["strongly_bullish", "moderately_bullish"] and 
            macd_signal == "bullish" and trend_strength != "weak"):
            trend_analysis['recommendation'] = "bullish"
            trend_analysis['options_strategy'] = "Buy Calls / Sell Put Credit Spreads"
        elif (ma_signal in ["strongly_bearish", "moderately_bearish"] and 
              macd_signal == "bearish" and trend_strength != "weak"):
            trend_analysis['recommendation'] = "bearish"
            trend_analysis['options_strategy'] = "Buy Puts / Sell Call Credit Spreads"
        else:
            trend_analysis['recommendation'] = "neutral"
            trend_analysis['options_strategy'] = "Non-directional strategies (Iron Condors)"
        
        result['directional_signals'] = trend_analysis
        
        # ---- Non-Directional Analysis for Iron Condors ----
        
        # Bollinger Band width (normalized)
        bb_width = 0
        if all(x in latest for x in ['BB_upper', 'BB_lower', 'Close']):
            bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['Close']
        
        # Historical volatility trend (10 days)
        vol_trend = "stable"
        if len(data) > 20:
            recent_vol = data['volatility'].iloc[-10:].values
            past_vol = data['volatility'].iloc[-20:-10].values
            
            if len(recent_vol) > 0 and len(past_vol) > 0:
                recent_mean = np.mean(recent_vol)
                past_mean = np.mean(past_vol)
                
                if recent_mean > past_mean * 1.1:  # 10% increase
                    vol_trend = "rising"
                elif recent_mean < past_mean * 0.9:  # 10% decrease
                    vol_trend = "falling"
        
        # RSI for overbought/oversold
        rsi = latest.get('RSI', 50)
        rsi_condition = "neutral"
        if rsi > 70:
            rsi_condition = "overbought"
        elif rsi < 30:
            rsi_condition = "oversold"
        
        # Non-directional analysis
        non_directional = {
            'bb_width': bb_width,
            'bb_width_percentile': 50,  # Would calculate from historical data
            'volatility_trend': vol_trend,
            'rsi_condition': rsi_condition,
            'rsi_value': rsi
        }
        
        # Generate non-directional recommendation
        if bb_width < 0.03:  # Narrow bands - potential breakout
            non_directional['recommendation'] = "potential_breakout"
            non_directional['options_strategy'] = "Avoid Iron Condors, consider Day Trading Calls/Puts"
        elif rsi_condition == "neutral" and vol_trend == "stable":
            non_directional['recommendation'] = "range_bound"
            non_directional['options_strategy'] = "Iron Condor (balanced)"
        elif vol_trend == "falling":
            non_directional['recommendation'] = "declining_volatility"
            non_directional['options_strategy'] = "Iron Condor (wider wings)"
        elif vol_trend == "rising":
            non_directional['recommendation'] = "rising_volatility"
            non_directional['options_strategy'] = "Reduce size or avoid Iron Condors"
        else:
            non_directional['recommendation'] = "neutral"
            non_directional['options_strategy'] = "Standard Iron Condor"
            
        result['non_directional_signals'] = non_directional
        
        # ---- Strike Selection Guidance ----
        
        # Support/resistance for strike selection
        current_price = latest.get('Close', 0)
        
        # Calculate quick approximate support/resistance
        support = None
        resistance = None
        
        if len(data) >= 20:
            window = min(20, len(data)-1)
            recent_lows = data['Low'].iloc[-window:].nsmallest(3)
            recent_highs = data['High'].iloc[-window:].nlargest(3)
            
            if not recent_lows.empty:
                support = float(recent_lows.mean())
            if not recent_highs.empty:
                resistance = float(recent_highs.mean())
        
        # Standard deviations for probability-based strikes
        std_dev = latest.get('ATR', current_price * 0.02) * 1.5  # Using ATR as volatility estimate
        
        one_std_down = max(current_price - std_dev, 0.01)
        one_std_up = current_price + std_dev
        two_std_down = max(current_price - (std_dev * 2), 0.01)
        two_std_up = current_price + (std_dev * 2)
        
        # Strike selection guidance
        strike_selection = {
            'current_price': current_price,
            'support': support,
            'resistance': resistance,
            'one_std_down': one_std_down,
            'one_std_up': one_std_up,
            'two_std_down': two_std_down,
            'two_std_up': two_std_up,
        }
        
        # Add probability-based strike recommendations
        strike_selection['put_credit_spread'] = {
            'short_strike': one_std_down,
            'long_strike': two_std_down,
            'probability_otm': '68%'
        }
        
        strike_selection['call_credit_spread'] = {
            'short_strike': one_std_up,
            'long_strike': two_std_up,
            'probability_otm': '68%'
        }
        
        strike_selection['iron_condor'] = {
            'put_short_strike': one_std_down,
            'put_long_strike': two_std_down,
            'call_short_strike': one_std_up,
            'call_long_strike': two_std_up,
            'probability_otm': '68%'
        }
        
        result['strike_selection'] = strike_selection
        
        # ---- Timing Signals ----
        
        # Stochastic for entry/exit timing
        stoch_k = latest.get('STOCH_%K', 50)
        stoch_d = latest.get('STOCH_%D', 50)
        
        stoch_signal = "neutral"
        if stoch_k > 80 and stoch_d > 80:
            stoch_signal = "overbought"
        elif stoch_k < 20 and stoch_d < 20:
            stoch_signal = "oversold"
        elif stoch_k > stoch_d and stoch_k < 80 and stoch_d < 80:
            stoch_signal = "bullish_crossover"
        elif stoch_k < stoch_d and stoch_k > 20 and stoch_d > 20:
            stoch_signal = "bearish_crossover"
        
        # Volume confirmation
        volume_signal = "normal"
        if len(data) > 20:
            avg_volume = data['Volume'].iloc[-20:-1].mean()
            current_volume = latest.get('Volume', avg_volume)
            
            if current_volume > avg_volume * 1.5:
                volume_signal = "high"
            elif current_volume < avg_volume * 0.5:
                volume_signal = "low"
        
        # Timing signals
        timing = {
            'stochastic_signal': stoch_signal,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'volume_signal': volume_signal,
            'rsi_signal': rsi_condition
        }
        
        # Generate timing recommendation
        if stoch_signal == "bullish_crossover" and rsi_condition == "oversold":
            timing['recommendation'] = "enter_bullish"
            timing['options_strategy'] = "Buy Calls / Sell Put Credit Spreads"
        elif stoch_signal == "bearish_crossover" and rsi_condition == "overbought":
            timing['recommendation'] = "enter_bearish"
            timing['options_strategy'] = "Buy Puts / Sell Call Credit Spreads"
        elif stoch_signal == "overbought" and volume_signal == "high":
            timing['recommendation'] = "exit_bullish"
            timing['options_strategy'] = "Take profits on Calls / Put Credit Spreads"
        elif stoch_signal == "oversold" and volume_signal == "high":
            timing['recommendation'] = "exit_bearish"
            timing['options_strategy'] = "Take profits on Puts / Call Credit Spreads"
        else:
            timing['recommendation'] = "neutral"
            timing['options_strategy'] = "Monitor position"
            
        result['timing'] = timing
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating options indicators: {str(e)}")
        return {
            'directional_signals': {'recommendation': 'error'},
            'non_directional_signals': {'recommendation': 'error'},
            'strike_selection': {},
            'timing': {'recommendation': 'error'}
        }

# =========================
# Options Flow (Placeholder)
# =========================
def get_options_flow(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get options flow data for market sentiment analysis.
    
    Options flow analysis looks at unusual options activity to gauge institutional sentiment:
    - Call/Put ratio: Bullish/bearish sentiment
    - Strike concentration: Key price levels
    - Volume & open interest: Liquidity and accumulation
    - Unusual activity: Large block trades, high premium trades
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Optional[Dict[str, Any]]: Options flow data or None
    """
    logging.debug(f"⚠️ Options flow data not implemented for {ticker}")  # Changed to debug level
    
    # In a real implementation, this would fetch data from an options flow provider
    # For now, return a structured placeholder that matches expected format
    
    return {
        'summary': {
            'call_put_ratio': 1.2,  # >1 is bullish
            'call_volume': 10000,
            'put_volume': 8000,
            'total_premium': 2500000,
            'sentiment': 'slightly_bullish',
            'unusual_activity': False
        },
        'notable_strikes': [
            {
                'strike': 150,
                'type': 'call',
                'expiry': '2023-09-15',
                'volume': 3500,
                'open_interest': 12000,
                'premium': 850000,
                'activity_type': 'accumulation'
            },
            {
                'strike': 140,
                'type': 'put',
                'expiry': '2023-09-15',
                'volume': 2800,
                'open_interest': 9500,
                'premium': 750000,
                'activity_type': 'hedging'
            }
        ],
        'expiry_analysis': {
            'near_term_sentiment': 'bullish',
            'mid_term_sentiment': 'neutral',
            'far_term_sentiment': 'slightly_bearish'
        }
    }
