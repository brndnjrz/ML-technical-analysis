"""
Core Data Management Module
==========================
Contains core data loading and processing functionality
"""

from .data_loader import fetch_stock_data, get_fundamental_data
from .data_pipeline import fetch_and_process_data

__all__ = [
    'fetch_stock_data',
    'get_fundamental_data', 
    'fetch_and_process_data'
]
