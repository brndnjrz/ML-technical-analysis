"""
Sidebar Quick Stats Component

This module provides the sidebar quick stats display functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def render_sidebar_quick_stats(data: pd.DataFrame, interval: str):
    """
    Render quick statistics in the sidebar
    
    Args:
        data: DataFrame with price data
        interval: Time interval of the data (e.g., "5m", "15m", "30m", "1h", "1d")
    """
    # Display quick price stats in sidebar
    st.sidebar.markdown("### 📊 Quick Stats")
    
    # Calculate basic stats
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    
    # Calculate percentage change
    pct_change = (current_price - prev_close) / prev_close * 100
    pct_change_color = "green" if pct_change >= 0 else "red"
    
    # Format the percentage change
    pct_change_str = f"{pct_change:.2f}%"
    
    # Create a custom metric display
    st.sidebar.markdown(f"""
    <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>
        <div style='color: #636c7a; font-size: 14px;'>Current Price</div>
        <div style='font-size: 24px; font-weight: bold;'>${current_price:.2f}</div>
        <div style='color: {pct_change_color}; font-size: 16px;'>{pct_change_str}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Volume comparison
    avg_volume = data['Volume'].mean()
    last_volume = data['Volume'].iloc[-1]
    vol_ratio = last_volume / avg_volume
    
    # Display volume
    st.sidebar.markdown(f"""
    <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-top: 10px;'>
        <div style='color: #636c7a; font-size: 14px;'>Volume</div>
        <div style='font-size: 20px; font-weight: bold;'>{last_volume:,.0f}</div>
        <div style='font-size: 14px;'>{vol_ratio:.1f}x avg volume</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate volatility
    if 'HV_20' in data.columns:
        volatility = data['HV_20'].iloc[-1]
    elif len(data) > 20:
        volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    else:
        volatility = data['Close'].pct_change().std() * np.sqrt(252)
    
    # Display volatility
    st.sidebar.markdown(f"""
    <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-top: 10px;'>
        <div style='color: #636c7a; font-size: 14px;'>Historical Volatility</div>
        <div style='font-size: 20px; font-weight: bold;'>{volatility:.2%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Range info
    high_52w = data['High'].max()
    low_52w = data['Low'].min()
    from_high = (current_price - high_52w) / high_52w * 100
    from_low = (current_price - low_52w) / low_52w * 100
    
    st.sidebar.markdown(f"""
    <div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-top: 10px;'>
        <div style='color: #636c7a; font-size: 14px;'>Range</div>
        <div style='font-size: 16px;'>High: ${high_52w:.2f}</div>
        <div style='font-size: 16px;'>Low: ${low_52w:.2f}</div>
        <div style='font-size: 14px; color: {"red" if from_high < 0 else "green"};'>{from_high:.1f}% from high</div>
    </div>
    """, unsafe_allow_html=True)
