# =========================
# Logging Configuration Module
# =========================
# This module provides centralized logging setup for the trading analysis application.
# It configures both console and file logging with different formats and levels.

import logging
import sys
from datetime import datetime

def setup_logging(level=logging.INFO, enable_file_logging=True):
    """
    MAIN FUNCTION: Sets up the entire logging system for the application
    
    This function configures:
    1. Console logging (what you see in terminal/Streamlit)
    2. File logging (saves all messages to a daily log file)
    3. Message formatting (how log messages look)
    4. Log levels (which messages to show/hide)
    
    Args:
        level: Controls what messages are shown in console
               - logging.DEBUG (10): Show everything (most verbose)
               - logging.INFO (20): Show info, warning, error (default)
               - logging.WARNING (30): Show only warnings and errors
               - logging.ERROR (40): Show only errors (least verbose)
        enable_file_logging: Whether to save logs to a file (True/False)
    
    Returns:
        root_logger: The main logger object that controls all logging
    """
    
    # STEP 1: Create message formatter for console output
    # This controls how log messages appear in your terminal/Streamlit app
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',  # Format: "14:30:45 - INFO - Your message"
        datefmt='%H:%M:%S'  # Time format: hours:minutes:seconds (short for console)
    )
    
    # STEP 2: Get the root logger (controls all logging in the app)
    # The root logger is the "master" logger that all other loggers inherit from
    root_logger = logging.getLogger()
    root_logger.setLevel(level)  # Set minimum level for ALL logging
    
    # STEP 3: Clear any existing handlers to prevent duplicate messages
    # This prevents logging messages from appearing multiple times
    # if setup_logging() is called more than once
    root_logger.handlers.clear()
    
    # STEP 4: Setup CONSOLE HANDLER (what you see in terminal/Streamlit)
    # StreamHandler sends log messages to sys.stdout (your console/terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)  # Apply the short format
    console_handler.setLevel(level)  # Console shows only messages at or above this level
    root_logger.addHandler(console_handler)  # Attach handler to root logger
    
    # STEP 5: Setup FILE HANDLER (saves logs to disk for later review)
    # This is optional and creates a daily log file for debugging/history
    if enable_file_logging:
        try:
            # Create filename with today's date: trading_analysis_20250811.log
            file_handler = logging.FileHandler(
                f'trading_analysis_{datetime.now().strftime("%Y%m%d")}.log', 
                mode='a'  # 'a' = append mode (add to existing file, don't overwrite)
            )
            
            # File formatter includes more detail (logger name) for debugging
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Includes logger name
            )
            file_handler.setFormatter(file_formatter)
            
            # IMPORTANT: File gets ALL messages (DEBUG level) regardless of console level
            # This means even if console only shows INFO+, file captures DEBUG messages too
            file_handler.setLevel(logging.DEBUG)  # File gets all messages for debugging
            root_logger.addHandler(file_handler)  # Attach file handler to root logger
            
        except Exception:
            # If file logging fails (permissions, disk space, etc.), continue without it
            # This prevents the entire app from crashing if file logging has issues
            pass  # Silently fail - app continues with just console logging
    
    return root_logger  # Return the configured logger

def set_log_level(level_name: str):
    """
    RUNTIME FUNCTION: Change logging level while the app is running
    
    This function allows users to change log verbosity without restarting the app.
    Used in Streamlit when user selects different log level from dropdown.
    
    HOW IT WORKS:
    1. Converts string name ("DEBUG") to logging constant (logging.DEBUG = 10)
    2. Updates the root logger level
    3. Updates console handler level to match
    4. File handler stays at DEBUG level (always captures everything)
    
    Args:
        level_name: String name of level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
                   Case-insensitive - 'debug', 'DEBUG', 'Debug' all work
    
    EXAMPLE USAGE:
        set_log_level("DEBUG")    # Show all messages including debug info
        set_log_level("ERROR")    # Show only error messages
    """
    # Convert string to logging constant using getattr()
    # getattr(logging, "DEBUG") returns logging.DEBUG (which equals 10)
    # If invalid level name, defaults to logging.INFO
    level = getattr(logging, level_name.upper(), logging.INFO)
    
    # Update root logger level (affects all loggers in the app)
    logging.getLogger().setLevel(level)
    
    # IMPORTANT: Also update console handler level
    # We need to find the console handler specifically and update its level
    # The file handler should stay at DEBUG level to capture everything
    for handler in logging.getLogger().handlers:
        # Check if this handler is the console handler (StreamHandler writing to stdout)
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)  # Update console handler to new level

def log_section(title: str):
    """
    UTILITY FUNCTION: Creates visual section separators in log output
    
    This function helps organize log output by creating clear visual breaks
    between different sections of your application (like data loading, analysis, etc.)
    
    Args:
        title: The section name to display
    
    OUTPUT EXAMPLE:
        ========== DATA LOADING ==========
    
    USAGE EXAMPLES:
        log_section("Starting Analysis")
        log_section("Processing Indicators")
        log_section("AI Analysis Complete")
    
    WHY USE THIS:
    - Makes log files easier to read
    - Helps identify different phases of processing
    - Useful for debugging to see where issues occur
    """
    # Creates a formatted section header with equals signs
    # f"{'='*10}" creates "=========="
    logging.info(f"{'='*10} {title} {'='*10}")


def log_performance(func_name: str, duration: float):
    """
    UTILITY FUNCTION: Logs performance/timing information for functions
    
    This function helps track how long different operations take,
    which is useful for:
    - Identifying slow functions that need optimization
    - Monitoring performance over time
    - Debugging timeout issues
    
    Args:
        func_name: Name of the function being timed
        duration: How long it took in seconds (float)
    
    OUTPUT EXAMPLE:
        ⏱️ calculate_indicators completed in 2.34s
        ⏱️ fetch_stock_data completed in 0.87s
    
    USAGE PATTERN:
        import time
        start_time = time.time()
        # ... do some work ...
        end_time = time.time()
        log_performance("my_function", end_time - start_time)
    
    NOTE: Uses DEBUG level so timing info only shows when debugging
    """
    # Uses DEBUG level because performance info is primarily for developers
    # Only visible when log level is set to DEBUG
    logging.debug(f"⏱️ {func_name} completed in {duration:.2f}s")


# =========================
# LOGGING LEVELS REFERENCE GUIDE
# =========================
# 
# LEVEL HIERARCHY (lowest to highest):
# 
# DEBUG (10)    - Detailed diagnostic info for developers
#               - Variable values, function entry/exit, detailed state info
#               - ONLY visible when log level = DEBUG
#               - Example: "🔍 Processing 150 data points with RSI period=14"
# 
# INFO (20)     - General application flow information
#               - What the app is doing at high level
#               - Visible when log level = INFO, DEBUG
#               - Example: "📊 Successfully loaded AAPL data for 2024-01-01 to 2024-12-31"
# 
# WARNING (30)  - Unexpected situations that don't stop the app
#               - Missing optional features, fallback behavior
#               - Visible when log level = WARNING, INFO, DEBUG
#               - Example: "⚠️ pandas_ta not available, using custom calculations"
# 
# ERROR (40)    - Serious problems that prevent operations from completing
#               - Network failures, invalid data, calculation errors
#               - ALWAYS visible (unless logging completely disabled)
#               - Example: "❌ Failed to fetch data for INVALID_TICKER: 404 Not Found"
# 
# CRITICAL (50) - System-level failures (not used in this app)
#               - App crashes, system out of memory, etc.
# 
# =========================
# HOW THIS MODULE IS USED IN THE APP
# =========================
# 
# 1. APP STARTUP (app.py):
#    setup_logging(level=logging.INFO, enable_file_logging=False)
#    - Sets default level to INFO (shows INFO, WARNING, ERROR)
#    - Disables file logging for Streamlit (keeps logs in console only)
# 
# 2. USER CONTROL (Streamlit UI):
#    User can change log level via dropdown:
#    - DEBUG: See everything (most verbose)
#    - INFO: Normal operation (default)
#    - WARNING: Only warnings and errors
#    - ERROR: Only error messages (quietest)
# 
# 3. LIBRARY SUPPRESSION:
#    logging.getLogger('kaleido').setLevel(logging.ERROR)
#    - Prevents noisy libraries from cluttering output
#    - Only shows their error messages, hides info/debug spam
# 
# 4. THROUGHOUT THE APP:
#    - logging.debug() for detailed diagnostic info
#    - logging.info() for normal operation messages  
#    - logging.warning() for unexpected but handleable situations
#    - logging.error() for failures that prevent operations
# 
# =========================
