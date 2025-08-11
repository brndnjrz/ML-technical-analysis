"""
Sidebar Indicator Selection Component

This module provides the sidebar indicator selection UI component.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional

def sidebar_indicator_selection(strategy_type, interval, data=None):
    """
    Display sidebar indicator selection options
    
    Args:
        strategy_type: Strategy type string
        interval: Time interval string ("1d", "1wk", "1mo")
        data: Optional DataFrame with price data
        
    Returns:
        List of selected indicators
    """
    # Categorized Indicators
    trend_indicators = [
        'SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'VWAP',
        'MACD', 'ADX', 'Supertrend', 'Parabolic SAR'
    ]
    
    momentum_indicators = [
        'RSI', 'Stochastic', 'CCI', 'Williams %R', 'Awesome Oscillator',
        'Momentum', 'ROC', 'TSI'
    ]
    
    volatility_indicators = [
        'Bollinger Bands', 'ATR', 'Standard Deviation', 
        'Keltner Channel', 'Donchian Channel'
    ]
    
    volume_indicators = [
        'OBV', 'Volume', 'Volume Profile', 'Money Flow Index',
        'Accumulation/Distribution', 'VWAP'
    ]
    
    options_indicators = [
        'Historical Volatility', 'Volatility Smile', 'Standard Deviation Moves',
        'IV Rank', 'Put-Call Ratio'
    ]
    
    # Default selections based on strategy type
    if strategy_type is not None and "Short-Term" in strategy_type:
        default_trend = ['SMA', 'EMA', 'MACD', 'VWAP']
        default_momentum = ['RSI', 'Stochastic']
        default_volatility = ['Bollinger Bands', 'ATR']
        default_volume = ['Volume', 'OBV']
    else:
        default_trend = ['SMA', 'EMA', 'MACD', 'ADX']
        default_momentum = ['RSI']
        default_volatility = ['Bollinger Bands']
        default_volume = ['Volume']
    
    # Add options indicators for options strategies
    default_options = []
    
    with st.sidebar.expander("📊 Technical Indicators", expanded=False):
        st.caption("Select indicators to display on chart")
        
        # Trend Indicators
        st.markdown("**Trend Indicators**")
        selected_trend = []
        for i, indicator in enumerate(trend_indicators):
            # Add unique key for each checkbox
            if st.checkbox(indicator, value=indicator in default_trend, key=f"trend_{i}"):
                selected_trend.append(indicator)
        
        # Momentum Indicators  
        st.markdown("**Momentum Indicators**")
        selected_momentum = []
        for i, indicator in enumerate(momentum_indicators):
            # Add unique key for each checkbox
            if st.checkbox(indicator, value=indicator in default_momentum, key=f"momentum_{i}"):
                selected_momentum.append(indicator)
        
        # Volatility Indicators
        st.markdown("**Volatility Indicators**")
        selected_volatility = []
        for i, indicator in enumerate(volatility_indicators):
            # Add unique key for each checkbox
            if st.checkbox(indicator, value=indicator in default_volatility, key=f"volatility_{i}"):
                selected_volatility.append(indicator)
                
        # Volume Indicators
        st.markdown("**Volume Indicators**")
        selected_volume = []
        for i, indicator in enumerate(volume_indicators):
            # Add unique key for each checkbox
            if st.checkbox(indicator, value=indicator in default_volume, key=f"volume_{i}"):
                selected_volume.append(indicator)
                
        # Options-specific indicators (shown only for options strategies)
        if "Options Trading Strategy" in st.session_state.get('analysis_type', ''):
            st.markdown("**Options Indicators**")
            selected_options = []
            for i, indicator in enumerate(options_indicators):
                # Add unique key for each checkbox
                if st.checkbox(indicator, value=indicator in default_options, key=f"options_{i}"):
                    selected_options.append(indicator)
        else:
            selected_options = []
    
    # Combine all selected indicators
    selected_indicators = selected_trend + selected_momentum + selected_volatility + selected_volume + selected_options
    
    return selected_indicators
