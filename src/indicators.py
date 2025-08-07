
# =========================
# Imports
# =========================
import pandas as pd
import pandas_ta as ta
import logging

# Set up logging
logger = logging.getLogger(__name__)

# =========================
# Indicator Calculation
# =========================
def calculate_indicators(data, timeframe="1d", strategy_type=None, selected_indicators=None):
    """
    Calculate technical indicators based on selected indicators.

    Args:
        data (pd.DataFrame): Stock price data
        timeframe (str): Timeframe for the data
        strategy_type (str): Strategy type (not used currently but needed for compatibility)
        selected_indicators (list): List of selected indicators to calculate

    Returns:
        pd.DataFrame: Data with calculated indicators
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
    data = data.copy()

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in stock data: {missing}")
    data = data.dropna(subset=required_columns)

    # Default indicators if none selected
    if selected_indicators is None:
        selected_indicators = ["RSI", "MACD", "SMA", "EMA", "VWAP", "Bollinger Bands", "ADX", "Stochastic", "OBV", "ATR"]
    selected_str = str(selected_indicators).lower()

    # Implied Volatility (IV)
    data['returns'] = data['Close'].pct_change().fillna(0)
    data['volatility'] = data['returns'].rolling(window=21).std().fillna(method='ffill').fillna(0) * (252 ** 0.5)

    # RSI
    if any(indicator.lower() in selected_str for indicator in ['rsi']):
        # Calculate both 14 and 21 period RSI
        data['RSI_14'] = ta.rsi(data['Close'], length=14).fillna(method='ffill').fillna(50)
        data['RSI_21'] = ta.rsi(data['Close'], length=21).fillna(method='ffill').fillna(50)
        # Store timeframe specific versions
        data[f'RSI_{timeframe}'] = data['RSI_14']
        data['RSI'] = data['RSI_14']  # Default RSI
        # Forward fill any remaining NaN values
        for col in ['RSI_14', 'RSI_21', f'RSI_{timeframe}', 'RSI']:
            data[col] = data[col].fillna(method='ffill').fillna(50)

    # MACD (always calculate)
    macd = ta.macd(data['Close'])
    # Store MACD values in multiple formats for compatibility
    data['MACD'] = macd['MACD_12_26_9'].fillna(method='ffill').fillna(0)
    data['MACD_Signal'] = macd['MACDs_12_26_9'].fillna(method='ffill').fillna(0)
    data['MACD_Hist'] = macd['MACDh_12_26_9'].fillna(method='ffill').fillna(0)
    # Store timeframe specific versions
    data[f'MACD_{timeframe}'] = data['MACD']
    data[f'MACD_Signal_{timeframe}'] = data['MACD_Signal']
    data[f'MACD_Hist_{timeframe}'] = data['MACD_Hist']
    # Store additional formats that might be needed
    data['MACD_Line'] = data['MACD']
    data['Signal_Line'] = data['MACD_Signal']
    data['MACD_Histogram'] = data['MACD_Hist']
    
    # Ensure all MACD columns are properly filled
    macd_cols = ['MACD', 'MACD_Signal', 'MACD_Hist', 
                 f'MACD_{timeframe}', f'MACD_Signal_{timeframe}', f'MACD_Hist_{timeframe}',
                 'MACD_Line', 'Signal_Line', 'MACD_Histogram']
    for col in macd_cols:
        if col in data.columns:
            data[col] = data[col].fillna(method='ffill').fillna(0)

    # SMA
    # Calculate SMAs and handle NaN values (always calculate these basic indicators)
    data['SMA_20'] = data['Close'].rolling(window=20).mean().fillna(method='ffill')
    data['SMA_50'] = data['Close'].rolling(window=50).mean().fillna(method='ffill')
    # Store timeframe specific versions
    data[f'SMA_20_{timeframe}'] = data['SMA_20']
    data[f'SMA_50_{timeframe}'] = data['SMA_50']
    # Add raw SMA columns without timeframe suffix
    data['SMA20'] = data['SMA_20']
    data['SMA50'] = data['SMA_50']
    
    # Ensure all SMA columns are properly filled
    sma_cols = ['SMA_20', 'SMA_50', f'SMA_20_{timeframe}', f'SMA_50_{timeframe}', 'SMA20', 'SMA50']
    for col in sma_cols:
        data[col] = data[col].fillna(method='ffill').fillna(data['Close'])

    # EMA (always calculate these basic indicators)
    # Calculate EMAs
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean().fillna(method='ffill')
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean().fillna(method='ffill')
    # Store timeframe specific versions
    data[f'EMA_20_{timeframe}'] = data['EMA_20']
    data[f'EMA_50_{timeframe}'] = data['EMA_50']
    
    # Ensure all EMA columns are properly filled
    ema_cols = ['EMA_20', 'EMA_50', f'EMA_20_{timeframe}', f'EMA_50_{timeframe}']
    for col in ema_cols:
        data[col] = data[col].fillna(method='ffill').fillna(data['Close'])

    # VWAP (always calculate)
    try:
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data["Volume"].cumsum()
        data['VWAP'] = data['VWAP'].fillna(method='ffill').fillna(data['Close'])
        data[f'VWAP_{timeframe}'] = data['VWAP']
    except Exception as e:
        logger.warning(f"Error calculating VWAP: {str(e)}")
        data['VWAP'] = data['Close']
        data[f'VWAP_{timeframe}'] = data['Close']

    # Bollinger Bands (always calculate)
    bb = ta.bbands(data['Close'], length=20, std=2)
    # Initialize with reasonable values for the first 20 periods
    data[f'BB_middle_{timeframe}'] = bb['BBM_20_2.0'].fillna(method='ffill').fillna(data['Close'])
    # Calculate standard deviation for initial upper/lower bands
    std = data['Close'].rolling(window=20).std().fillna(method='ffill').fillna(data['Close'].std())
    data[f'BB_upper_{timeframe}'] = bb['BBU_20_2.0'].fillna(data[f'BB_middle_{timeframe}'] + 2 * std)
    data[f'BB_lower_{timeframe}'] = bb['BBL_20_2.0'].fillna(data[f'BB_middle_{timeframe}'] - 2 * std)
    
    # Store standard versions without timeframe
    data['BB_upper'] = data[f'BB_upper_{timeframe}']
    data['BB_lower'] = data[f'BB_lower_{timeframe}']
    data['BB_middle'] = data[f'BB_middle_{timeframe}']
    
    # Ensure all BB columns are properly filled
    bb_cols = ['BB_upper', 'BB_lower', 'BB_middle',
               f'BB_upper_{timeframe}', f'BB_lower_{timeframe}', f'BB_middle_{timeframe}']
    for col in bb_cols:
        data[col] = data[col].fillna(method='ffill')

    # ADX (always calculate)
    adx_result = ta.adx(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx_result['ADX_14'].fillna(method='ffill').fillna(25)  # Fill with neutral value
    data[f'ADX_{timeframe}'] = data['ADX']
    
    # Ensure ADX columns are properly filled
    for col in ['ADX', f'ADX_{timeframe}']:
        data[col] = data[col].fillna(method='ffill').fillna(25)

    # Stochastic
    if any(indicator.lower() in selected_str for indicator in ['stoch']):
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        data[f'STOCH_%K_{timeframe}'] = stoch['STOCHk_14_3_3']
        data[f'STOCH_%D_{timeframe}'] = stoch['STOCHd_14_3_3']

    # OBV
    if any(indicator.lower() in selected_str for indicator in ['obv']):
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        data[f'OBV_{timeframe}'] = data['OBV']

    # ATR (always calculate)
    # Calculate ATR and handle initial NaN values
    atr = ta.atr(data['High'], data['Low'], data['Close'])
    # Calculate initial ATR value using first available data
    initial_atr = (data['High'].iloc[0] - data['Low'].iloc[0]) if not atr.empty else 0
    data['ATR'] = atr.fillna(method='ffill').fillna(initial_atr)
    data[f'ATR_{timeframe}'] = data['ATR']
    
    # Ensure ATR columns are properly filled
    for col in ['ATR', f'ATR_{timeframe}']:
        if col in data.columns:
            data[col] = data[col].fillna(method='ffill').fillna(initial_atr)

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
