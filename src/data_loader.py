import yfinance as yf

# Function to Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    # Fetches historical data between selected dates
    return stock.history(start=start_date, end=end_date, auto_adjust=False)