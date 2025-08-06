
# =========================
# Imports
# =========================
import pandas as pd
import pandas_ta as ta

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
    data['returns'] = data['Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=21).std() * (252 ** 0.5)

    # RSI
    if any(indicator.lower() in selected_str for indicator in ['rsi']):
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data[f'RSI_{timeframe}'] = data['RSI']

    # MACD
    if any(indicator.lower() in selected_str for indicator in ['macd']):
        macd = ta.macd(data['Close'])
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_Signal'] = macd['MACDs_12_26_9']
        data[f'MACD_{timeframe}'] = data['MACD']
        data[f'MACD_Signal_{timeframe}'] = data['MACD_Signal']

    # SMA
    if any(indicator.lower() in selected_str for indicator in ['sma', 'moving']):
        data[f'SMA_20_{timeframe}'] = data['Close'].rolling(window=20).mean()
        data[f'SMA_50_{timeframe}'] = data['Close'].rolling(window=50).mean()

    # EMA
    if any(indicator.lower() in selected_str for indicator in ['ema', 'moving']):
        data[f'EMA_20_{timeframe}'] = data['Close'].ewm(span=20, adjust=False).mean()
        data[f'EMA_50_{timeframe}'] = data['Close'].ewm(span=50, adjust=False).mean()

    # VWAP
    if any(indicator.lower() in selected_str for indicator in ['vwap']):
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data["Volume"].cumsum()

    # Bollinger Bands
    if any(indicator.lower() in selected_str for indicator in ['bollinger', 'bb']):
        bb = ta.bbands(data['Close'], length=20, std=2)
        data[f'BB_upper_{timeframe}'] = bb['BBU_20_2.0']
        data[f'BB_lower_{timeframe}'] = bb['BBL_20_2.0']
        data[f'BB_middle_{timeframe}'] = bb['BBM_20_2.0']

    # ADX
    if any(indicator.lower() in selected_str for indicator in ['adx']):
        adx_result = ta.adx(data['High'], data['Low'], data['Close'])
        data['ADX'] = adx_result['ADX_14']
        data[f'ADX_{timeframe}'] = data['ADX']

    # Stochastic
    if any(indicator.lower() in selected_str for indicator in ['stoch']):
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        data[f'STOCH_%K_{timeframe}'] = stoch['STOCHk_14_3_3']
        data[f'STOCH_%D_{timeframe}'] = stoch['STOCHd_14_3_3']

    # OBV
    if any(indicator.lower() in selected_str for indicator in ['obv']):
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        data[f'OBV_{timeframe}'] = data['OBV']

    # ATR
    if any(indicator.lower() in selected_str for indicator in ['atr']):
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'])
        data[f'ATR_{timeframe}'] = data['ATR']

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
