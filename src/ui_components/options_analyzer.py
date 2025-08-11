"""
Options Analyzer Tab

This module provides the Options Analyzer functionality for the Finance app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import plotly.graph_objects as go

def display_options_analyzer(ticker: str, data: pd.DataFrame, options_data: Optional[Dict] = None):
    """
    Display the Options Analyzer tab with strategy selection and visualization
    
    Args:
        ticker: Stock symbol
        data: Historical price data with indicators
        options_data: Optional dictionary containing options chain data
    """
    st.header(f"Options Strategy Analyzer - {ticker}")
    
    # Handle potential connection issues with external data providers
    with st.status("Loading Options Analyzer...", expanded=False) as status:
        # Try to import our custom options strategy selector
        try:
            from src.ui_components.options_strategy_selector import display_options_strategy_selector
            
            # Show a warning about potential timeout issues
            st.info("📊 Options analysis may take a moment. If you see timeout errors, the analysis will continue with limited data.")
            
            try:
                # Display options strategy selector
                status.update(label="Loading strategy selector...", state="running")
                strategy_config = display_options_strategy_selector(data, ticker)
                
                # Validate strategy config
                if not isinstance(strategy_config, dict):
                    st.warning("Strategy configuration data is invalid. Using default settings.")
                    strategy_config = {
                        "strategy": "Long Call",
                        "parameters": {
                            "strike": data['Close'].iloc[-1] * 1.05,
                            "contracts": 1
                        }
                    }
                    
                # Store the strategy config
                if 'strategy_configs' not in st.session_state:
                    st.session_state['strategy_configs'] = {}
                st.session_state['strategy_configs'][ticker] = strategy_config
                
                # Get current price safely
                try:
                    current_price = float(data['Close'].iloc[-1])
                except (IndexError, ValueError, TypeError):
                    current_price = 100  # Fallback value if we can't get the price
                
                # Display payoff diagram
                status.update(label="Generating payoff diagram...", state="running")
                try:
                    display_strategy_payoff(strategy_config, current_price)
                except Exception as payoff_error:
                    st.error(f"Could not generate payoff diagram: {payoff_error}")
                    st.info("Using simplified strategy visualization.")
                    # Display a simple placeholder chart if the payoff diagram fails
                    st.line_chart(data['Close'].tail(30))
                    
                status.update(label="Options analyzer loaded successfully!", state="complete")
            except Exception as e:
                st.error(f"Error in options analyzer: {str(e)}")
                st.info("Using limited data for options analysis. Some features may be unavailable.")
                # Try to continue with limited functionality
                
        except ImportError:
            st.error("Options strategy selector component not available")
            status.update(label="Failed to load Options Analyzer", state="error")

def display_strategy_payoff(strategy_config: Dict[str, Any], current_price: float):
    """
    Display a payoff diagram for the selected options strategy
    
    Args:
        strategy_config: Dictionary with strategy configuration
        current_price: Current stock price
    """
    st.subheader("Strategy Payoff Diagram")
    
    # Validate input data
    if not isinstance(strategy_config, dict):
        st.error("Invalid strategy configuration")
        return
        
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        st.error("Invalid current price")
        current_price = 100  # Fallback to a reasonable default
    
    # Get strategy details with safe defaults
    strategy_type = strategy_config.get('strategy_type', 'Long Call')
    category = strategy_config.get('category', 'Directional')
    parameters = strategy_config.get('parameters', {})
    
    # Create price range for x-axis
    # Default to +/- 20% if no std dev strikes available
    if parameters.get('std_dev_strikes'):
        strikes = parameters['std_dev_strikes']
        price_min = min(strikes.values()) * 0.9
        price_max = max(strikes.values()) * 1.1
    else:
        price_min = current_price * 0.8
        price_max = current_price * 1.2
    
    # Generate price points
    prices = np.linspace(price_min, price_max, 100)
    
    # Calculate payoff based on strategy type
    payoffs = np.zeros_like(prices)
    
    # Number of contracts (each represents 100 shares)
    contracts = parameters.get('contracts', 1)
    contract_multiplier = contracts * 100
    
    # For strategies with width between strikes
    width = parameters.get('width', 5)
    
    # Calculate break-even points and payoffs based on strategy type
    if strategy_type == "Long Call":
        strike = current_price
        premium = current_price * 0.03  # Estimated premium
        
        payoffs = np.maximum(prices - strike, 0) - premium
        break_even = strike + premium
        
    elif strategy_type == "Long Put":
        strike = current_price
        premium = current_price * 0.03  # Estimated premium
        
        payoffs = np.maximum(strike - prices, 0) - premium
        break_even = strike - premium
        
    elif strategy_type == "Bull Call Spread":
        lower_strike = current_price
        upper_strike = current_price + width
        premium = width * 0.4  # Estimated net premium
        
        payoffs = np.minimum(np.maximum(prices - lower_strike, 0), upper_strike - lower_strike) - premium
        break_even = lower_strike + premium
        
    elif strategy_type == "Bear Put Spread":
        upper_strike = current_price + width/2
        lower_strike = current_price - width/2
        premium = width * 0.4  # Estimated net premium
        
        payoffs = np.minimum(np.maximum(upper_strike - prices, 0), upper_strike - lower_strike) - premium
        break_even = upper_strike - premium
        
    elif strategy_type == "Iron Condor":
        middle = current_price
        put_spread_width = width
        call_spread_width = width
        
        put_short_strike = middle - width/2
        put_long_strike = put_short_strike - put_spread_width
        call_short_strike = middle + width/2
        call_long_strike = call_short_strike + call_spread_width
        
        credit = width * 0.2  # Estimated credit
        
        # Calculate Iron Condor payoff
        put_spread = np.minimum(np.maximum(put_short_strike - prices, 0), put_spread_width) - put_spread_width
        call_spread = np.minimum(np.maximum(prices - call_short_strike, 0), call_spread_width) - call_spread_width
        payoffs = credit + put_spread + call_spread
        
        break_even_lower = put_short_strike - credit
        break_even_upper = call_short_strike + credit
        break_even = [break_even_lower, break_even_upper]
    
    # Multiply by contract size
    payoffs = payoffs * contract_multiplier
    
    # Create payoff diagram
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(go.Scatter(
        x=prices,
        y=payoffs,
        mode='lines',
        name=strategy_type,
        line=dict(color='blue', width=3)
    ))
    
    # Add break-even points
    if isinstance(break_even, list):
        for be in break_even:
            fig.add_vline(x=be, line_dash="dash", line_color="green", annotation_text=f"Break-even: ${be:.2f}")
    else:
        fig.add_vline(x=break_even, line_dash="dash", line_color="green", annotation_text=f"Break-even: ${break_even:.2f}")
    
    # Add current price line
    fig.add_vline(x=current_price, line_color="red", annotation_text=f"Current: ${current_price:.2f}")
    
    # Add zero line
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    
    # Update layout
    fig.update_layout(
        title=f"{strategy_type} Payoff at Expiration",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified"
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Display strategy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Profit", f"${np.max(payoffs):.2f}")
    
    with col2:
        st.metric("Max Loss", f"${np.min(payoffs):.2f}")
    
    with col3:
        risk_reward = abs(np.min(payoffs) / np.max(payoffs)) if np.max(payoffs) != 0 else 0
        st.metric("Risk/Reward Ratio", f"{risk_reward:.2f}")
        
    # Display strategy explanation
    st.subheader("Strategy Overview")
    
    strategy_explanations = {
        "Long Call": """
            **Long Call Strategy**
            
            This is a bullish strategy where you purchase a call option, giving you the right to buy the stock at the strike price.
            
            **When to use:**
            - You expect the stock to rise significantly
            - You want to limit risk while maintaining unlimited upside potential
            - You prefer defined risk with leverage
            
            **Risks:**
            - Time decay works against this position
            - Requires significant price movement to overcome premium paid
        """,
        "Long Put": """
            **Long Put Strategy**
            
            This is a bearish strategy where you purchase a put option, giving you the right to sell the stock at the strike price.
            
            **When to use:**
            - You expect the stock to decline significantly
            - You want protection against downside risk (as a hedge)
            - You prefer defined risk with leverage
            
            **Risks:**
            - Time decay works against this position
            - Requires significant price movement to overcome premium paid
        """,
        "Iron Condor": """
            **Iron Condor Strategy**
            
            This is a neutral strategy consisting of a bull put spread and a bear call spread, creating a range where you profit if the stock stays between the short strikes.
            
            **When to use:**
            - You expect the stock to trade within a range
            - Implied volatility is high (selling premium is favorable)
            - You want to generate income with defined risk
            
            **Risks:**
            - Large price moves in either direction can lead to losses
            - Early assignment risk on short options
            - Multiple legs increase transaction costs
        """
    }
    
    # Display explanation for the selected strategy
    if strategy_type in strategy_explanations:
        st.markdown(strategy_explanations[strategy_type])
    else:
        st.markdown(f"**{strategy_type}** is an options strategy that can be used in a {category.lower()} market scenario.")
