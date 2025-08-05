# import yfinance as yf

# # Function to Fetch Stock Data
# def fetch_stock_data(ticker, start_date=None, end_date=None, interval="1d"):
#     valid_intraday = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
#     # period = "7d" if interval in valid_intraday else None

#     try:
#         stock = yf.Ticker(ticker)

#         if interval in valid_intraday:
#             # Limit period to avoid throttling (1m allows max 7d)
#             period = "7d"
#             df = stock.history(period=period, interval=interval)
#             # return stock.history(period=period, interval=interval)
#         else:
#             # Fetches historical data between selected dates
#             # return stock.history(start=start_date, end=end_date, auto_adjust=False)
#             df = stock.history(start=start_date, end=end_date, interval=interval)
#             # return stock.history(start=start_date, end=end_date, interval=interval)

#         # Check if data was returned 
#         if df is None or df.empty:
#             print(f"[Warning] No data returned for {ticker} with interval {interval}")
#             return pd.DataFrame()

#         # Normalize column names to match expected format
#         df.rename(columns=lambda x: x.capitalize(), inplace=True)

#         # Ensure required columns exist 
#         required_cols = ["Open", "High", "Low", "Close", "Volume"]
#         if not all(col in df.columns for col in required_cols):
#             print(f"[Error] Missing columns in data: {df.columns.tolist()}")
#             return pd.DataFrame()
        
#         return df

#     except Exception as e:
#         print(f"[Error] Failed to fetch data for {ticker}: {e}")









import yfinance as yf
import pandas as pd  # ← ADD THIS IMPORT

# Function to Fetch Stock Data
def fetch_stock_data(ticker, start_date=None, end_date=None, interval="1d"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        interval (str): Data interval (1d, 1h, etc.)
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns, or None if failed
    """
    valid_intraday = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]

    try:
        stock = yf.Ticker(ticker)
        
        # Validate ticker exists
        info = stock.info
        if not info or 'symbol' not in info:
            print(f"[Error] Invalid ticker symbol: {ticker}")
            return None

        if interval in valid_intraday:
            # Limit period to avoid throttling (1m allows max 7d)
            period = "7d"
            df = stock.history(period=period, interval=interval)
        else:
            # Fetches historical data between selected dates
            df = stock.history(start=start_date, end=end_date, interval=interval)

        # Check if data was returned 
        if df is None or df.empty:
            print(f"[Warning] No data returned for {ticker} with interval {interval}")
            return None  # ← CHANGED: Return None instead of empty DataFrame

        # Debug: Print original columns
        print(f"[Debug] Original columns: {df.columns.tolist()}")

        # Fix column names - yfinance returns proper case already
        # But let's ensure consistency
        column_mapping = {
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adj_Close'  # Handle adjusted close
        }
        
        # Rename columns to ensure consistency
        df = df.rename(columns=column_mapping)
        
        # Alternative: If columns are still not right, force the names
        if len(df.columns) >= 5:
            # yfinance typically returns: Open, High, Low, Close, Volume, (sometimes Adj Close)
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if list(df.columns[:5]) != expected_cols:
                # Force rename the first 5 columns
                new_columns = expected_cols + list(df.columns[5:])
                df.columns = new_columns

        # Ensure required columns exist 
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"[Error] Missing columns in data: {missing_cols}")
            print(f"[Error] Available columns: {df.columns.tolist()}")
            return None  # ← CHANGED: Return None instead of empty DataFrame
        
        # Remove any rows with NaN values in critical columns
        df = df.dropna(subset=required_cols)
        
        if df.empty:
            print(f"[Warning] No valid data after cleaning for {ticker}")
            return None
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        print(f"[Success] Fetched {len(df)} rows for {ticker}")
        print(f"[Success] Final columns: {df.columns.tolist()}")
        
        return df

    except Exception as e:
        print(f"[Error] Failed to fetch data for {ticker}: {e}")
        import traceback
        traceback.print_exc()  # Print full error for debugging
        return None  # ← ADD THIS: Always return None on error

# Additional helper function for testing
def test_data_fetch(ticker="AAPL"):
    """Test function to verify data fetching works"""
    import datetime
    
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

if __name__ == "__main__":
    # Test the function when run directly
    test_data_fetch()
