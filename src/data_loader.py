import yfinance as yf

# Function to Fetch Stock Data
def fetch_stock_data(ticker, start_date=None, end_date=None, interval="1d"):
    valid_intraday = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    # period = "7d" if interval in valid_intraday else None

    try:
        stock = yf.Ticker(ticker)

        if interval in valid_intraday:
            # Limit period to avoid throttling (1m allows max 7d)
            period = "7d"
            df = stock.history(period=period, interval=interval)
            # return stock.history(period=period, interval=interval)
        else:
            # Fetches historical data between selected dates
            # return stock.history(start=start_date, end=end_date, auto_adjust=False)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            # return stock.history(start=start_date, end=end_date, interval=interval)

        # Check if data was returned 
        if df is None or df.empty:
            print(f"[Warning] No data returned for {ticker} with interval {interval}")
            return pd.DataFrame()

        # Normalize column names to match expected format
        df.rename(columns=lambda x: x.capitalize(), inplace=True)

        # Ensure required columns exist 
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            print(f"[Error] Missing columns in data: {df.columns.tolist()}")
            return pd.DataFrame()
        
        return df

    except Exception as e:
        print(f"[Error] Failed to fetch data for {ticker}: {e}")