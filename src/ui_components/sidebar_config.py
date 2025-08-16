"""
Sidebar Configuration Components

This module provides the sidebar configuration UI components.
"""

import streamlit as st
import datetime
from datetime import timedelta
from typing import Dict, Any, Tuple, List, Optional
from ..utils.config import DEFAULT_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE
from ..trading_strategies import get_strategy_names

def sidebar_config(config=None):
    """
    Display sidebar configuration options for the app
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Tuple of (ticker, start_date, end_date, interval, analysis_type, 
                 strategy_type, options_strategy, options_priority)
    """
    st.sidebar.title("Trading Assistant")
    
    ticker = st.sidebar.text_input("Enter Stock Symbol", DEFAULT_TICKER)
    
    # Date range selection
    start_date = st.sidebar.date_input("Start Date", DEFAULT_START_DATE.date())
    end_date = st.sidebar.date_input("End Date", DEFAULT_END_DATE.date())
    
    interval_options = {
        "5m": "5min (Scalping)",
        "15m": "15min (Day Trading)",
        "30m": "30min (Short-term)",
        "1h": "1hour (Swing)",
        "1d": "Daily (Long-term)"
    }
    interval = st.sidebar.selectbox(
        "Select Timeframe:",
        options=list(interval_options.keys()),
        format_func=lambda x: interval_options[x],
        index=1
    )
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Options Trading Strategy", "Stock Buy/Hold/Sell", "Advanced Analysis (AI Ensemble)"]
    )
    strategy_type = None
    options_strategy = None
    
    # Set strategy type for all analysis types
    if analysis_type == "Options Trading Strategy":
        strategy_type = st.sidebar.selectbox(
            "Trading Timeframe:",
            ["Short-Term (1-7 days)", "Medium-Term (1-3 weeks)"],
            index=1
        )
        
        # AI will automatically select the optimal strategy
        st.sidebar.info("🤖 AI will select the optimal options strategy based on current market conditions")
        options_strategy = "AI-Selected"
        
        # Show the types of strategies that may be considered
        with st.sidebar.expander("What strategies will be considered?"):
            strategy_names = get_strategy_names()
            st.markdown("The AI system will analyze market conditions to select the optimal strategy from:")
            
            for strategy in strategy_names:
                st.markdown(f"• **{strategy}**")
            
            st.markdown("""
            Each strategy includes multiple timeframes:
            - **Short-Term (1–7 days)**: Quick moves and scalping
            - **Medium-Term (1–3 weeks)**: Trend-following and income generation
            - **Intraday**: Minutes to hours for day trading
            """)
            
        # The AI now uses a comprehensive approach combining all strike selection methods
        st.sidebar.info("🧠 AI will analyze strikes using multiple methods: Standard Deviation, Technical Levels, and Delta-Based approaches for optimal selection")
        strike_method = "AI-Comprehensive"
    elif analysis_type == "Stock Buy/Hold/Sell":
        strategy_type = st.sidebar.selectbox(
            "Trading Timeframe:",
            ["Short-Term (1-7 days)", "Medium-Term (1-3 weeks)"],
            index=1
        )
        if "Short-Term" in strategy_type:
            options_strategy = "Day Trading"
        elif "Medium-Term" in strategy_type:
            options_strategy = "Swing Trading"
    
    # Options Priority Checkbox
    options_priority = False
    if analysis_type == "Options Trading Strategy":
        options_priority = True
    elif analysis_type == "Advanced Analysis (AI Ensemble)":
        options_priority = st.sidebar.checkbox(
            "Prioritize Options Strategies", 
            value=True,
            help="Focus on options-based strategies (calls, puts, spreads) rather than stock positions"
        )
        
    return ticker, start_date, end_date, interval, analysis_type, strategy_type, options_strategy, options_priority
