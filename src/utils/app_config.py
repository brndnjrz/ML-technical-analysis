"""Application configuration constants and settings."""

# Session State Keys
class SessionKeys:
    STOCK_DATA = "stock_data"
    LEVELS = "levels"
    OPTIONS_DATA = "options_data"
    ACTIVE_INDICATORS = "active_indicators"
    ANALYSIS_TYPE = "analysis_type"
    AI_ANALYSIS_RESULT = "ai_analysis_result"
    AI_ANALYSIS_RUNNING = "ai_analysis_running"
    RUN_ANALYSIS = "run_analysis"

# Technical Analysis Thresholds
class TechnicalThresholds:
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_NEUTRAL_LOWER = 30
    RSI_NEUTRAL_UPPER = 70
    
    MACD_WEAK_SIGNAL_THRESHOLD = 0.1
    ADX_STRONG_TREND = 25
    
    IV_RANK_LOW = 30
    IV_RANK_HIGH = 60
    
    ATR_SMALL_RATIO = 0.015
    ATR_MODERATE_RATIO = 0.03
    ATR_MULTIPLIER_STOP = 2
    ATR_MULTIPLIER_TARGET = 3
    
    VIX_LOW = 15
    VIX_MODERATE = 25

# Analysis Progress Steps
class ProgressSteps:
    DATA_FETCH_START = 25
    DATA_FETCH_COMPLETE = 75
    DATA_FETCH_FINAL = 100
    
    PREDICTION = 20
    CHART_PREP = 40
    AI_ANALYSIS = 60
    COMPLETION = 100

# UI Configuration
class UIConfig:
    PAGE_TITLE = "AI Technical Analysis"
    APP_TITLE = "Technical Stock Analysis Dashboard"
    SIDEBAR_HEADER = "⚙️ Configuration"
    
    # Chart dimensions
    CHART_HEIGHT = 1000
    
    # Default values
    DEFAULT_CONFIDENCE = 0.5
    MIN_DATASET_SIZE = 20

# Chart Base Columns (for filtering indicators)
BASE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

# Indicator Groups for UI Display
INDICATOR_GROUPS = {
    "Trend Indicators": ["SMA", "EMA", "MACD", "ADX"],
    "Momentum Indicators": ["RSI", "STOCH", "CCI"],
    "Volatility Indicators": ["BBands", "ATR", "Standard Deviation"],
    "Volume Indicators": ["OBV", "VWAP", "Volume"]
}

# Log Level Options
LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR"]

# Analysis Types
class AnalysisTypes:
    STOCK_TRADING = "Stock Buy/Hold/Sell"
    OPTIONS_TRADING = "Options Trading Strategy"

# Strategy Types
class StrategyTypes:
    SHORT_TERM = "Short-Term"
    LONG_TERM = "Long-Term"

# Timeframe Mapping
TIMEFRAME_MAPPING = {
    "1m": "intraday",
    "5m": "intraday", 
    "15m": "intraday",
    "1h": "1-7d",
    "1d": "2-4w"
}

def get_user_timeframe(strategy_type, interval):
    """Determine user timeframe based on strategy type and interval."""
    if "intraday" in strategy_type.lower() or interval in ["1m", "5m", "15m"]:
        return "intraday"
    elif "short" in strategy_type.lower():
        return "1-7d"
    else:
        return "2-4w"