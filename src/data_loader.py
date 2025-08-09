
# =========================
# Imports
# =========================
import yfinance as yf
import datetime
import pandas as pd
import logging 

# =========================
# Constants
# =========================
VALID_INTRADAY_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
COLUMN_MAPPING = {
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume',
    'Adj Close': 'Adj_Close'  # Handle adjusted close
}

def get_fundamental_data(ticker):
    """
    Fetch fundamental data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary containing fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get earnings data
        try:
            earnings = stock.earnings
            eps_ttm = earnings['EPS'].sum() if not earnings.empty else None
        except:
            eps_ttm = None
        
        # Calculate metrics
        fundamentals = {
            'EPS': info.get('trailingEPS') or eps_ttm,
            'Revenue Growth': info.get('revenueGrowth', info.get('earningsGrowth')),
            'Profit Margin': info.get('profitMargins'),
            'P/E Ratio': info.get('trailingPE'),
            'Market Cap': info.get('marketCap'),
            'Volume': info.get('volume'),
            'Average Volume': info.get('averageVolume'),
            'Forward P/E': info.get('forwardPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Beta': info.get('beta'),
            'Dividend Yield': info.get('dividendYield', 0)
        }
        
        # Convert ratios to percentages where appropriate
        if fundamentals['Revenue Growth'] is not None:
            fundamentals['Revenue Growth'] = fundamentals['Revenue Growth'] * 100
        if fundamentals['Profit Margin'] is not None:
            fundamentals['Profit Margin'] = fundamentals['Profit Margin'] * 100
        if fundamentals['Dividend Yield'] is not None:
            fundamentals['Dividend Yield'] = fundamentals['Dividend Yield'] * 100
            
        return fundamentals
        
    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return {}

# =========================
# Main Data Fetching Function
# =========================
def fetch_stock_data(ticker, start_date=None, end_date=None, interval="1d"):
    """
    Fetch stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        interval (str): Data interval (1d, 1h, etc.)

    Returns:
        pd.DataFrame: Stock data with OHLCV columns, or None if failed
    """
    try:
        stock = yf.Ticker(ticker)

        # Validate ticker exists
        info = stock.info
        if not info or 'symbol' not in info:
            print(f"[Error] Invalid ticker symbol: {ticker}")
            return None

        # Fetch fundamental data
        fundamentals = get_fundamental_data(ticker)
        # print(f"[Info] Fundamental data: {fundamentals}")
        logging.info(f"Fundamental data for {ticker}: {fundamentals}")

        # Fetch data based on interval
        if interval in VALID_INTRADAY_INTERVALS:
            # Limit period to avoid throttling (1m allows max 7d)
            period = "7d"
            df = stock.history(period=period, interval=interval)
        else:
            df = stock.history(start=start_date, end=end_date, interval=interval)

        # Check if data was returned
        if df is None or df.empty:
            print(f"[Warning] No data returned for {ticker} with interval {interval}")
            return None

        # print(f"[Debug] Original columns: {df.columns.tolist()}")
        logging.debug(f"Original columns: {df.columns.tolist()}")

        # Standardize column names
        df = df.rename(columns=COLUMN_MAPPING)

        # Force rename if columns are not as expected
        if len(df.columns) >= 5:
            expected_cols = REQUIRED_COLS
            if list(df.columns[:5]) != expected_cols:
                new_columns = expected_cols + list(df.columns[5:])
                df.columns = new_columns

        # Ensure required columns exist
        missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing_cols:
            print(f"[Error] Missing columns in data: {missing_cols}")
            print(f"[Error] Available columns: {df.columns.tolist()}")
            return None

        # Remove rows with NaN in critical columns
        df = df.dropna(subset=REQUIRED_COLS)
        if df.empty:
            print(f"[Warning] No valid data after cleaning for {ticker}")
            return None

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # print(f"[Success] Fetched {len(df)} rows for {ticker}")
        # print(f"[Success] Final columns: {df.columns.tolist()}")
        logging.info(f"Fetched {len(df)} rows for {ticker}")
        logging.info(f"Final columns: {df.columns.tolist()}")
        return df

    except Exception as e:
        print(f"[Error] Failed to fetch data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

# =========================
# Helper/Test Function
# =========================
def test_data_fetch(ticker="AAPL"):
    """
    Test function to verify data fetching works.
    """
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=30)
    print(f"Testing data fetch for {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date, "1d")
    if data is not None:
        print(f"✅ Success! Got {len(data)} rows")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print("\nFirst few rows:")
        print(data.head())
        return True
    else:
        print("❌ Failed to fetch data")
        return False

# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    test_data_fetch()
