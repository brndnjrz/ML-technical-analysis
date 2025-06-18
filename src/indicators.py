import pandas as pd
import pandas_ta as ta

# Function to Calculate Technical Indicators
def calculate_indicators(data, timeframe="1d"):
    data = data.copy()

    # ✅ Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in stock data: {missing}")


    # Clean missing candles
    data = data.dropna(subset=required_columns)
    # data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Implied Volatility (IV)
    data['returns'] = data['Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=21).std() * (252 ** 0.5)

    # Relative Strength Index (RSI)
    data[f'RSI_{timeframe}'] = ta.rsi(data['Close'], length=14)

    # MACD
    macd = ta.macd(data['Close'])
    data[f'MACD_{timeframe}'] = macd['MACD_12_26_9']
    data[f'MACD_Signal_{timeframe}'] = macd['MACDs_12_26_9']

    # Moving Averages (Simple Moving Average and Exponential Moving Average)
    data[f'SMA_20_{timeframe}'] = data['Close'].rolling(window=20).mean()
    data[f'SMA_50_{timeframe}'] = data['Close'].rolling(window=50).mean()
    data[f'EMA_20_{timeframe}'] = data['Close'].ewm(span=20, adjust=False).mean()
    data[f'EMA_50_{timeframe}'] = data['Close'].ewm(span=50, adjust=False).mean()

    # VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data["Volume"].cumsum()

    # 20-Day Bollinger Bands
    bb = ta.bbands(data['Close'], length=20, std=2)
    data[f'BB_upper_{timeframe}'] = bb['BBU_20_2.0']
    data[f'BB_lower_{timeframe}'] = bb['BBL_20_2.0']
    data[f'BB_middle_{timeframe}'] = bb['BBM_20_2.0']

    # ADX (Trend Strength)
    data[f'ADX_{timeframe}'] = ta.adx(data['High'], data['Low'], data['Close'])['ADX_14']

    # Stochastic Oscillator
    stoch = ta.stoch(data['High'], data['Low'], data['Close'])
    data[f'STOCH_%K_{timeframe}'] = stoch['STOCHk_14_3_3']
    data[f'STOCH_%D_{timeframe}'] = stoch['STOCHd_14_3_3']

    # OBV (On-Balance Volume)
    data[f'OBV_{timeframe}'] = ta.obv(data['Close'], data['Volume'])

    # ATR (Average True Range)
    data[f'ATR_{timeframe}'] = ta.atr(data['High'], data['Low'], data['Close'])

    # Return Data
    return data

# Function to Summarize Technical Indicators in Plain English 
def summarize_indicators(df):
    summary = []
    latest = df.iloc[-1]

    rsi = latest.get("RSI", None)
    if pd.notnull(rsi):
        if rsi > 70:
            summary.append(f"RSI is {rsi:.1f}, indicating the stock is overbought.")
        elif rsi < 30:
            summary.append(f"RSI is {rsi:.1f}, indicating the stock is oversold.")
        else:
            summary.append(f"RSI is {rsi:.1f}, suggesting neutral momentum.")
    
    macd = latest.get("MACD", None)
    signal = latest.get("MACD_Signal", None)
    if pd.notnull(macd) and pd.notnull(signal):
        if macd > signal:
            summary.append(f"MACD is above the signal line ({macd:.2f} > {signal:.2f}), showing bullish momentum.")
        elif macd < signal:
            summary.append(f"MACD is below the signal line ({macd:.2f} < {signal:.2f}), showing bearish momentum.")
        else:
            summary.append(f"MACD and signal line are equal, indicating a neutral trend.")

    close_price = latest.get("Close", None)
    upper_band = latest.get("BB_upper", None)
    lower_band = latest.get("BB_lower", None)
    if pd.notnull(close_price) and pd.notnull(upper_band) and pd.notnull(lower_band):
        if close_price >= upper_band:
            summary.append(f"Price (${close_price:.2f}) is near or above the upper Bollinger Band, suggesting overbought conditions.")
        elif close_price <= lower_band: 
            summary.append(f"Price (${close_price:.2f}) is near or below the lower Bollinger Band, suggesting oversold conditions.")
        else:
            summary.append(f"Price (${close_price:.2f}) is within Bollinger Band, suggesting consolidation.")

    iv = latest.get("volatility", None)
    if pd.notnull(iv):
        iv_percent = iv * 100
        if iv_percent > 30:
            summary.append(f"Volatility is {iv_percent:.1f}%, which is high and may favor premium-selling strategies.")
        elif iv_percent < 20:
            summary.append(f"Volatility is {iv_percent:.1f}%, which is low and may favor debit spreads.")
        else:
            summary.append(f"Volatility is {iv_percent:.1f}%, considered moderate.")

    volume = latest.get("Volume", None)
    avg_volume = df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df else None
    if pd.notnull(volume) and pd.notnull(avg_volume):
        if volume > 1.5 * avg_volume:
            summary.append(f"Volume is significantly above average ({volume:.0f} vs {avg_volume:.0f}, indicating strong interests.)")
        elif volume < 0.5 * avg_volume:
            summary.append(f"Volume is much lower than average ({volume:.0f} vs {avg_volume:.0f}, indicating weak interests.)")
        else:
            summary.append(f"Volume is near the 20-day average ({volume:.0f} vs {avg_volume:.0f}.)")

    adx = latest.get("ADX_1d", None)
    if pd.notnull(adx):
        if adx > 25:
            summary.append(f"ADX is {adx:.1f}, indicating a strong trend.")
        elif adx < 20:
            summary.append(f"ADX is {adx:.1f}, indicating a weak or ranging market.")

    obv = latest.get("OBV_1d", None)
    if pd.notnull(obv):
        summary.append(f"OBV is {obv:.0f}, showing momentum based on volume flow.")

    atr = latest.get("ATR_1d", None)
    if pd.notnull(atr):
        summary.append(f"ATR is {atr:.2f}, suggesting current volatility level.")

    stoch_k = latest.get("STOCH_%K_1d", None)
    stoch_d = latest.get("STOCH_%D_1d", None)
    if pd.notnull(stoch_k) and pd.notnull(stoch_d):
        if stoch_k > 80:
            summary.append(f"Stochastic Oscillator (%K={stoch_k:.1f}) indicates overbought conditions.")
        elif stoch_k < 20:
            summary.append(f"Stochastic Oscillator (%K={stoch_k:.1f}) indicates oversold conditions.")

    return "\n".join(summary)

# Detect Support and Resistance Levels
def detect_support_resistance(df, window=20, tolerance=0.01):
    """
    Detect support and resistance zones using local minima and maxima in price history.
    Returns: dict with 'support' and 'resistance' lists of levels.
    """
    supports = []
    resistances = []

    for i in range(window, len(df) - window):
        # Support: local minima
        local_lows = df['Low'].iloc[i - window:i + window]
        if df['Low'].iloc[i] == local_lows.min():
            level = df['Low'].iloc[i]
            if not any(abs(level - s) / s < tolerance for s in supports):
                supports.append(level)

        #  Resistance: local maxima
        local_highs = df['High'].iloc[i - window:i + window]
        if df['High'].iloc[i] == local_highs.max():
            level = df['High'].iloc[i]
            if not any(abs(level - r) / r < tolerance for r in resistances):
                resistances.append(level)

    supports = sorted(supports)
    resistances = sorted(resistances)
    return {
        "support": supports[-3:], # return last 3 relevant levels
        "resistance": resistances[-3:]
    }