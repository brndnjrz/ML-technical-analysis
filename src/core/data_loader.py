
# =========================
# Imports
# =========================
import yfinance as yf
import datetime
import pandas as pd
import logging 

# =========================
# Logging Helper
# =========================
def log_data_info(ticker: str, message_type: str, **kwargs):
    """Centralized logging for data operations"""
    if message_type == "fundamentals":
        fundamentals = kwargs.get('fundamentals', {})
        fundamental_summary = []
        for key, value in fundamentals.items():
            if value is not None and value != 'N/A':
                if key in ['Revenue Growth', 'Profit Margin', 'Dividend Yield']:
                    fundamental_summary.append(f"{key}: {value:.1f}%")
                elif key in ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Beta']:
                    fundamental_summary.append(f"{key}: {value:.2f}")
                elif key == 'Market Cap':
                    fundamental_summary.append(f"{key}: ${value/1e9:.1f}B")
                elif key in ['Volume', 'Average Volume']:
                    fundamental_summary.append(f"{key}: {value:,}")
                else:
                    fundamental_summary.append(f"{key}: {value}")
        
        if fundamental_summary:
            logging.info(f"📊 {ticker} | {' | '.join(fundamental_summary[:3])}")
        else:
            logging.info(f"📊 {ticker} | No fundamental data available")
            
    elif message_type == "data_fetched":
        rows = kwargs.get('rows', 0)
        interval = kwargs.get('interval', 'unknown')
        logging.info(f"✅ {ticker} | Fetched {rows} {interval} candles")
        
    elif message_type == "error":
        error = kwargs.get('error', 'Unknown error')
        logging.error(f"❌ {ticker} | {error}") 

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
    Fetch fundamental data for a given ticker
    """
    try:
        # Create ticker with timeout parameter
        stock = yf.Ticker(ticker)
        # Set a shorter timeout for the API request
        info = stock.fast_info  # Using fast_info instead of info to reduce timeout issues
        
        # Get basic info with fast_info and try to get more details with minimal requests
        # Only fetch more detailed info if absolutely needed and with a short timeout
        detailed_info = {}
        try:
            # Try to get some additional info with a strict timeout
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            session = requests.Session()
            retries = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            session.mount('http://', HTTPAdapter(max_retries=retries))
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            # Try to get stock info safely without using private methods
            try:
                info_data = stock.info
                detailed_info = info_data if isinstance(info_data, dict) else {}
            except Exception as info_err:
                print(f"Error getting stock.info: {info_err}")
                detailed_info = {}
        except Exception as e:
            print(f"Skipping detailed info due to: {e}")
            detailed_info = {}
            
        # Initialize safe values
        fast_info_values = {}
        try:
            # Safely access fast_info attributes
            if hasattr(stock, 'fast_info'):
                fast_info = stock.fast_info
                # Use a safe access pattern for each attribute
                if hasattr(fast_info, 'trailing_pe'):
                    fast_info_values['trailing_pe'] = fast_info.trailing_pe if fast_info.trailing_pe is not None else 0.0
                if hasattr(fast_info, 'market_cap'):
                    fast_info_values['market_cap'] = fast_info.market_cap if fast_info.market_cap is not None else 0.0
                if hasattr(fast_info, 'last_volume'):
                    fast_info_values['last_volume'] = fast_info.last_volume if fast_info.last_volume is not None else 0
                if hasattr(fast_info, 'three_month_average_daily_volume'):
                    fast_info_values['three_month_average_daily_volume'] = fast_info.three_month_average_daily_volume if fast_info.three_month_average_daily_volume is not None else 0
                if hasattr(fast_info, 'beta'):
                    fast_info_values['beta'] = fast_info.beta if fast_info.beta is not None else 1.0
                if hasattr(fast_info, 'dividend_yield'):
                    fast_info_values['dividend_yield'] = fast_info.dividend_yield if fast_info.dividend_yield is not None else 0.0
        except Exception as e:
            print(f"Error accessing fast_info attributes: {e}")
        
        # Calculate metrics with default values and fallbacks
        # Make sure detailed_info is a dictionary
        if not isinstance(detailed_info, dict):
            detailed_info = {}
            
        fundamentals = {
            'EPS': detailed_info.get('trailingEPS', 0.0),
            'Revenue Growth': detailed_info.get('revenueGrowth', 0.0),
            'Profit Margin': detailed_info.get('profitMargins', 0.0),
            'P/E Ratio': fast_info_values.get('trailing_pe', 0.0),
            'Market Cap': fast_info_values.get('market_cap', 0.0),
            'Volume': fast_info_values.get('last_volume', 0),
            'Average Volume': fast_info_values.get('three_month_average_daily_volume', 0),
            'Forward P/E': detailed_info.get('forwardPE', 0.0),
            'PEG Ratio': detailed_info.get('pegRatio', 0.0),
            'Beta': fast_info_values.get('beta', 1.0),
            'Dividend Yield': fast_info_values.get('dividend_yield', 0.0)
        }
        
        # Convert ratios to percentages where appropriate
        try:
            fundamentals['Revenue Growth'] = float(fundamentals['Revenue Growth']) * 100
        except (ValueError, TypeError):
            fundamentals['Revenue Growth'] = 0.0
            
        try:
            fundamentals['Profit Margin'] = float(fundamentals['Profit Margin']) * 100
        except (ValueError, TypeError):
            fundamentals['Profit Margin'] = 0.0
            
        try:
            fundamentals['Dividend Yield'] = float(fundamentals['Dividend Yield']) * 100
        except (ValueError, TypeError):
            fundamentals['Dividend Yield'] = 0.0
        
        return fundamentals
        
    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return {}

# =========================
# Main Data Fetching Function
# =========================
def fetch_stock_data(ticker, start_date=None, end_date=None, interval="15m"):
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
        log_data_info(ticker, "fundamentals", fundamentals=fundamentals)

        # Fetch data based on interval
        if interval in VALID_INTRADAY_INTERVALS:
            # Limit period to avoid throttling (1m allows max 7d)
            period = "7d"
            df = stock.history(period=period, interval=interval)
        else:
            df = stock.history(start=start_date, end=end_date, interval=interval)
        
        # Store ticker symbol as attribute for later reference
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.attrs['ticker'] = ticker

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
        log_data_info(ticker, "data_fetched", rows=len(df), interval=interval)
        logging.debug(f"📋 Columns: {', '.join(df.columns.tolist())}")
        return df

    except Exception as e:
        log_data_info(ticker, "error", error=f"Failed to fetch data: {e}")
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
    data = fetch_stock_data(ticker, start_date, end_date, "15m")
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
