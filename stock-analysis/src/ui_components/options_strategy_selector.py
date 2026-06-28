import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from ..trading_strategies import strategies_data, get_strategy_names, get_strategy_by_name, get_timeframes_for_strategy

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
    
    # Get available strategies from the new format
    available_strategies = get_strategy_names()
    
    # Create a selectbox for strategy selection
    selected_strategy_name = st.selectbox("Select Trading Strategy", available_strategies)
    
    # Get the selected strategy data
    selected_strategy = get_strategy_by_name(selected_strategy_name)
    
    if selected_strategy:
        # Get available timeframes for the selected strategy
        available_timeframes = get_timeframes_for_strategy(selected_strategy_name)
        
        if available_timeframes:
            # Timeframe selection
            selected_timeframe = st.selectbox("Select Timeframe", available_timeframes)
            
            # Display strategy details for the selected timeframe
            timeframe_data = selected_strategy['Timeframes'][selected_timeframe]
            
            # Show strategy information
            st.markdown(f"### {selected_strategy_name} - {selected_timeframe}")
            
            # Best use case
            st.markdown("**Best Use:**")
            st.write(timeframe_data.get('Best_Use', 'No description available'))
            
            # Key indicators
            st.markdown("**Key Indicators:**")
            key_indicators = timeframe_data.get('Key_Indicators', [])
            for indicator in key_indicators:
                st.write(f"• {indicator}")
            
            # Advanced tips
            with st.expander("💡 Advanced Tips"):
                advanced_tips = timeframe_data.get('Advanced_Tips', [])
                for tip in advanced_tips:
                    st.write(f"• {tip}")
    
    # Legacy strategy categories for backwards compatibility
    st.markdown("---")
    st.markdown("### Alternative Strategy Categories")
    
    strategy_categories = [
        "Bullish Strategies", 
        "Bearish Strategies",
        "Neutral Strategies",
        "Volatility Strategies"
    ]
    
    category = st.selectbox("Legacy Strategy Category", strategy_categories)
    
    # Define strategies for each category (only using the 6 approved strategies)
    legacy_strategies = {
        "Bullish Strategies": [
            "Covered Calls",
            "Swing Trading"
        ],
        "Bearish Strategies": [
            "Cash-Secured Puts",
            "Credit Spreads"
        ],
        "Neutral Strategies": [
            "Iron Condors",
            "Credit Spreads"
        ],
        "Volatility Strategies": [
            "Day Trading Calls/Puts",
            "Iron Condors"
        ]
    }
    
    # Strategy selector
    legacy_strategy = st.selectbox("Select Legacy Strategy", legacy_strategies[category])
    
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
    if "Spread" in legacy_strategy or "Iron" in legacy_strategy:
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider("Width Between Strikes ($)", 1, 20, 5)
        with col2:
            contracts = st.number_input("Number of Contracts", 1, 100, 1)
    elif "Long" in legacy_strategy:
        contracts = st.number_input("Number of Contracts", 1, 100, 1)
    
    # Risk analysis
    st.markdown("### Risk Analysis")
    
    # Different metrics based on strategy type
    if category in ["Bullish Strategies", "Bearish Strategies"]:
        max_loss = contracts * 100 * width if "Spread" in legacy_strategy else contracts * 100 * current_price * 0.05
        max_profit = "Unlimited" if "Long" in legacy_strategy and "Spread" not in legacy_strategy else contracts * 100 * width
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Loss", f"${max_loss:.2f}")
        with col2:
            if isinstance(max_profit, str):
                st.metric("Max Profit", max_profit)
            else:
                st.metric("Max Profit", f"${max_profit:.2f}")
    elif "Iron Condor" in legacy_strategy:
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
        "strategy_type": selected_strategy_name if selected_strategy else legacy_strategy,
        "category": category,
        "expiration": expiration,
        "current_price": current_price,
        "timeframe": selected_timeframe if 'selected_timeframe' in locals() else None,
        "strategy_data": selected_strategy if selected_strategy else None,
        "parameters": {
            "width": width if "Spread" in legacy_strategy or "Iron" in legacy_strategy else None,
            "contracts": contracts,
            "std_dev_strikes": std_dev_strikes if 'HV_20' in volatility else None
        }
    }
