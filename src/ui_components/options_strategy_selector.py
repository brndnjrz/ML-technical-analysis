import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def display_options_strategy_selector(data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Display an advanced options strategy selector UI component
    
    Args:
        data: DataFrame with historical price data and indicators
        ticker: Stock symbol
    
    Returns:
        Dictionary with options strategy configuration
    """
    st.subheader(f"Options Strategy Builder for {ticker}")
    
    # Calculate current price and volatility metrics
    current_price = data['Close'].iloc[-1]
    
    # Get volatility metrics if available
    volatility = {}
    for col in ['HV_10', 'HV_20', 'HV_60', 'VOL_SKEW', 'IC_SUITABILITY']:
        if col in data.columns:
            volatility[col] = data[col].iloc[-1]
    
    # Display current price and volatility metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    if 'HV_20' in volatility:
        with col2:
            st.metric("20-Day HV", f"{volatility['HV_20']:.1%}")
    
    if 'IC_SUITABILITY' in volatility:
        with col3:
            st.metric("Iron Condor Score", f"{volatility['IC_SUITABILITY']:.1f}/100")
    
    # Strategy selection
    st.markdown("### Strategy Selection")
    
    strategy_categories = [
        "Bullish Strategies",
        "Bearish Strategies",
        "Neutral Strategies",
        "Volatility Strategies"
    ]
    
    category = st.selectbox("Strategy Category", strategy_categories)
    
    # Define strategies for each category
    strategies = {
        "Bullish Strategies": [
            "Long Call",
            "Bull Call Spread",
            "Bull Put Spread",
            "Call Debit Spread",
            "LEAPS Call"
        ],
        "Bearish Strategies": [
            "Long Put",
            "Bear Put Spread", 
            "Bear Call Spread",
            "Put Debit Spread",
            "LEAPS Put"
        ],
        "Neutral Strategies": [
            "Iron Condor",
            "Iron Butterfly",
            "Short Straddle",
            "Calendar Spread"
        ],
        "Volatility Strategies": [
            "Long Straddle",
            "Long Strangle",
            "Butterfly Spread"
        ]
    }
    
    # Strategy selector
    strategy = st.selectbox("Select Strategy", strategies[category])
    
    # Standard deviation based strikes
    if 'HV_20' in volatility:
        hv = volatility['HV_20']
        std_dev_strikes = {
            "1-Std Dev Up": round(current_price * (1 + hv), 2),
            "1-Std Dev Down": round(current_price * (1 - hv), 2),
            "2-Std Dev Up": round(current_price * (1 + hv * 2), 2),
            "2-Std Dev Down": round(current_price * (1 - hv * 2), 2)
        }
        
        st.markdown("### Potential Strike Selection")
        
        # Display standard deviation based strikes
        st.markdown("**Standard Deviation Based Strikes**")
        cols = st.columns(4)
        for i, (label, price) in enumerate(std_dev_strikes.items()):
            with cols[i]:
                st.metric(label, f"${price}")
    
    # Display strategy guidance and parameters
    st.markdown("### Strategy Parameters")
    
    expiration_options = ["1 week", "2 weeks", "1 month", "2 months", "3 months"]
    expiration = st.selectbox("Expiration", expiration_options)
    
    # Strategy specific parameters
    if "Spread" in strategy or "Iron" in strategy:
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider("Width Between Strikes ($)", 1, 20, 5)
        with col2:
            contracts = st.number_input("Number of Contracts", 1, 100, 1)
    elif "Long" in strategy:
        contracts = st.number_input("Number of Contracts", 1, 100, 1)
    
    # Risk analysis
    st.markdown("### Risk Analysis")
    
    # Different metrics based on strategy type
    if category in ["Bullish Strategies", "Bearish Strategies"]:
        max_loss = contracts * 100 * width if "Spread" in strategy else contracts * 100 * current_price * 0.05
        max_profit = "Unlimited" if "Long" in strategy and "Spread" not in strategy else contracts * 100 * width
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Loss", f"${max_loss:.2f}")
        with col2:
            if isinstance(max_profit, str):
                st.metric("Max Profit", max_profit)
            else:
                st.metric("Max Profit", f"${max_profit:.2f}")
    elif "Iron Condor" in strategy:
        max_loss = contracts * 100 * width
        max_profit = contracts * 100 * (width * 0.4)  # Typical credit is ~40% of width
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Loss", f"${max_loss:.2f}")
        with col2:
            st.metric("Max Profit", f"${max_profit:.2f}")
        with col3:
            st.metric("Risk/Reward", f"{max_loss/max_profit:.1f}:1")
    
    # Return strategy configuration
    return {
        "strategy_type": strategy,
        "category": category,
        "expiration": expiration,
        "current_price": current_price,
        "parameters": {
            "width": width if "Spread" in strategy or "Iron" in strategy else None,
            "contracts": contracts,
            "std_dev_strikes": std_dev_strikes if 'HV_20' in volatility else None
        }
    }
