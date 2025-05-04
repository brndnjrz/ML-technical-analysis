import pandas as pd
import pandas_ta as ta

# Function to Calculate Technical Indicators
def calculate_indicators(data):
    # Implied Volatility (IV)
    data['returns'] = data['Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=21).std() * (252 ** 0.5)

    # Relative Strength Index (RSI)
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # MACD
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_Signal'] = macd['MACDs_12_26_9']

    # Moving Averages (Simple Moving Average and Exponential Moving Average)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data["Volume"].cumsum()

    # 20-Day Bollinger Bands
    bb = ta.bbands(data['Close'], length=20, std=2)
    data['BB_upper'] = bb['BBU_20_2.0']
    data['BB_lower'] = bb['BBL_20_2.0']
    data['BB_middle'] = bb['BBM_20_2.0']

    # Return Data
    return data