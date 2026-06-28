import pandas as pd

# Default settings for sidebar input
DEFAULT_TICKER = "AAPL"
# Start date is 3 months before today
DEFAULT_START_DATE = pd.Timestamp.today() - pd.DateOffset(months=3)
DEFAULT_END_DATE = pd.Timestamp.today()