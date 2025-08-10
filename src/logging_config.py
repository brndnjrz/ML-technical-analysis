# =========================
# Logging Configuration
# =========================
import logging
import sys
from datetime import datetime

def setup_logging(level=logging.INFO, enable_file_logging=True):
    """
    Setup centralized logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_file_logging: Whether to log to file
    """
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'  # Shorter time format for console
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if enable_file_logging:
        try:
            file_handler = logging.FileHandler(
                f'trading_analysis_{datetime.now().strftime("%Y%m%d")}.log', 
                mode='a'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            root_logger.addHandler(file_handler)
        except Exception:
            pass  # If file logging fails, continue without it
    
    return root_logger

def set_log_level(level_name: str):
    """
    Change logging level at runtime
    
    Args:
        level_name: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    
    # Update console handler level
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)

def log_section(title: str):
    """Log a section separator"""
    logging.info(f"{'='*10} {title} {'='*10}")

def log_performance(func_name: str, duration: float):
    """Log performance metrics"""
    logging.debug(f"⏱️ {func_name} completed in {duration:.2f}s")
