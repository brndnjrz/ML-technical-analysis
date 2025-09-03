"""
Options Analysis Module for Finance Application
"""

import pandas as pd
import logging
import time
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from ..utils.options_strategy_cheatsheet import OPTIONS_STRATEGY_CHEATSHEET, get_analyst_cheatsheet_markdown

# Setup logger
logger = logging.getLogger(__name__)

def fetch_options_data(ticker: str, use_cache: bool = True, days_to_expiry: int = 30) -> Dict[str, Any]:
    """
    Fetch options chain data for a given ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY')
        use_cache: Whether to use cached data if available and recent
        days_to_expiry: Target days to expiration for focusing analysis
        
    Returns:
        Dictionary containing:
        - expirations: List of expiration dates
        - iv_data: Implied volatility metrics
        - calls: Call options for selected expiration
        - puts: Put options for selected expiration
        - atm_strike: At-the-money strike price
    """
    logger.info(f"Fetching options data for {ticker.upper()}...")
    
    try:
        # Use a simple mock implementation for development/testing
        # In production, replace with actual API calls to a data provider
        
        # Mock current stock price (normally you'd get this from a real API)
        import yfinance as yf
        try:
            ticker_data = yf.Ticker(ticker)
            current_price = ticker_data.history(period="1d")["Close"].iloc[-1]
        except Exception as e:
            logger.warning(f"Error getting current price from yfinance: {e}")
            # Fallback mock price
            current_price = 100.0 if ticker.upper() != "SPY" else 450.0
            
        # Generate fake expiration dates
        today = datetime.now()
        expirations = []
        for i in range(1, 6):  # Weekly expirations
            expiry = today + timedelta(days=i*7)
            # Format as YYYY-MM-DD
            expirations.append(expiry.strftime("%Y-%m-%d"))
        
        for i in range(1, 4):  # Monthly expirations
            expiry = today + timedelta(days=i*30)
            # Format as YYYY-MM-DD
            expirations.append(expiry.strftime("%Y-%m-%d"))
            
        # Select target expiration (closest to requested days to expiry)
        target_date = today + timedelta(days=days_to_expiry)
        closest_expiry = min(expirations, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days))
        
        # Calculate ATM strike based on current price
        atm_strike = round(current_price / 5) * 5  # Round to nearest $5
        
        # Generate mock options chain data
        iv_data = {
            "iv_rank": 45.2,  # IV Rank (0-100)
            "iv_percentile": 38.7,  # IV Percentile (0-100)
            "hv_30": 22.5,  # 30-day historical volatility
            "iv_30": 25.8,  # 30-day implied volatility
            "iv_term_structure": [
                {"days": 7, "iv": 24.2},
                {"days": 14, "iv": 25.1},
                {"days": 30, "iv": 25.8},
                {"days": 60, "iv": 26.9}
            ]
        }
        
        # Generate mock call options
        calls = []
        for i in range(-5, 6):  # 11 strikes centered around ATM
            strike = atm_strike + (i * 5)
            calls.append({
                "strike": strike,
                "expiry": closest_expiry,
                "bid": max(0.05, round((current_price - strike + 5) * 0.1, 2)),
                "ask": max(0.10, round((current_price - strike + 5) * 0.1 + 0.15, 2)),
                "iv": max(10, iv_data["iv_30"] - (i * 1.2)),  # IV smile
                "delta": max(0.01, min(0.99, 0.5 + (0.1 * i))),
                "gamma": round(0.05 - abs(i * 0.008), 3),
                "theta": round(-0.03 - abs(i * 0.005), 3),
                "volume": int(100 * (5 - abs(i))),
                "open_interest": int(500 * (5 - abs(i)))
            })
            
        # Generate mock put options
        puts = []
        for i in range(-5, 6):  # 11 strikes centered around ATM
            strike = atm_strike + (i * 5)
            puts.append({
                "strike": strike,
                "expiry": closest_expiry,
                "bid": max(0.05, round((strike - current_price + 5) * 0.1, 2)),
                "ask": max(0.10, round((strike - current_price + 5) * 0.1 + 0.15, 2)),
                "iv": max(10, iv_data["iv_30"] - (i * 1.2)),  # IV smile
                "delta": max(-0.99, min(-0.01, -0.5 - (0.1 * i))),
                "gamma": round(0.05 - abs(i * 0.008), 3),
                "theta": round(-0.03 - abs(i * 0.005), 3),
                "volume": int(80 * (5 - abs(i))),
                "open_interest": int(400 * (5 - abs(i)))
            })
            
        # Return compiled options data
        options_data = {
            "ticker": ticker.upper(),
            "price": current_price,
            "timestamp": datetime.now().isoformat(),
            "expirations": expirations,
            "selected_expiry": closest_expiry,
            "days_to_expiry": (datetime.strptime(closest_expiry, "%Y-%m-%d") - today).days,
            "iv_data": iv_data,
            "calls": calls,
            "puts": puts,
            "atm_strike": atm_strike
        }
        
        return options_data
        
    except Exception as e:
        logger.error(f"Error fetching options data: {str(e)}")
        return None

def analyze_options_chain(
    data: pd.DataFrame, 
    ticker: str, 
    current_price: float, 
    short_term_trend: str, 
    iv_rank: float
) -> Dict[str, Any]:
    """
    Analyze options data using professional strategy framework to provide recommendations
    
    Args:
        data: DataFrame with historical price and indicator data
        ticker: Stock symbol
        current_price: Current price of the stock
        short_term_trend: Detected short-term trend ('uptrend', 'downtrend', 'sideways')
        iv_rank: Implied Volatility Rank (0-100)
    Returns:
        Dictionary with options analysis results and recommendations
    """
    try:
        # Get recent price data and indicators
        recent_data = data.tail(30)  
        
        # Calculate ATR for volatility assessment
        if 'atr' in recent_data.columns:
            atr_value = recent_data['atr'].mean()
        elif 'Close' in recent_data.columns:
            atr_value = recent_data['Close'].std()
        elif 'close' in recent_data.columns:
            atr_value = recent_data['close'].std()
        else:
            # Fallback: use std of first numeric column if available
            numeric_cols = recent_data.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                atr_value = recent_data[numeric_cols[0]].std()
            else:
                atr_value = 1.0  # Safe fallback
        
        # Get RSI for momentum assessment
        rsi_value = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else 50
        
        # Get ADX if available
        adx_value = recent_data['adx'].iloc[-1] if 'adx' in recent_data.columns else 15
        
        # Determine market regime based only on short term trend
        if short_term_trend == "uptrend":
            market_regime = "Bullish"
            trend = "bullish"
        elif short_term_trend == "downtrend":
            market_regime = "Bearish"
            trend = "bearish"
        else:
            market_regime = "Neutral"
            trend = "neutral"
        
        # Determine volatility environment
        volatility_env = "high_volatility" if iv_rank > 50 else "low_volatility"
        
        # Determine price pattern based on indicators
        if rsi_value > 70:
            price_pattern = "overbought"
        elif rsi_value < 30:
            price_pattern = "oversold"
        elif adx_value > 25:
            price_pattern = "trending"
        else:
            price_pattern = "rangebound"
        
        # Check step 5 strategy selection matrix from the cheatsheet
        strategy_key = f"{trend}_{volatility_env}"
        
        # Extract strategy from the cheatsheet
        strategy_matrix = OPTIONS_STRATEGY_CHEATSHEET["step5_strategy_selection"]["matrix"]
        
        if strategy_key in strategy_matrix:
            strategy_name = strategy_matrix[strategy_key]["strategy"]
            strategy_rationale = strategy_matrix[strategy_key]["rationale"]
            strategy_description = strategy_matrix[strategy_key].get("Description", "No description available.")
        else:
            # Default strategy if not found in the matrix
            if trend == "bullish":
                strategy_name = "Long Calls"
                strategy_rationale = "Default bullish strategy"
                strategy_description = "Buy calls when expecting a strong upward move. Profit if the stock rises above the strike."
            elif trend == "bearish":
                strategy_name = "Long Puts"
                strategy_rationale = "Default bearish strategy"
                strategy_description = "Buy puts when expecting a significant downward move. Profit if the stock falls below the strike."
            else:
                strategy_name = "Iron Condor"
                strategy_rationale = "Default neutral strategy for range-bound markets"
                strategy_description = "Sell iron condors when expecting little movement. Profit if the stock stays between the short strikes."
        
        # Apply risk management from step 6
        risk_rules = OPTIONS_STRATEGY_CHEATSHEET["step6_risk_management"]["rules"]
        
        # Build the response with strategy details
        options_analysis = {
            "ticker": ticker,
            "current_price": current_price,
            "market_conditions": {
                "short_term_trend": short_term_trend,
                "market_regime": market_regime,
                "iv_rank": iv_rank,
                "rsi": rsi_value,
                "atr": atr_value,
                "adx": adx_value if 'adx' in recent_data.columns else None
            },
            "strategy_recommendation": {
                "name": strategy_name,
                "trend_direction": trend,
                "volatility_environment": volatility_env.replace("_", " "),
                "price_pattern": price_pattern,
                "rationale": strategy_rationale,
                "Description": strategy_description
            },
            "risk_management": {
                "rules": risk_rules,
                "position_sizing": "Max 1-2% of portfolio risk per trade",
                "recommended_stop": f"${round(current_price * 0.95, 2)}" if trend == "bullish" else f"${round(current_price * 1.05, 2)}"
            },
            "analysis_process": {
                "step1": "Identify Market Trend",
                "step2": "Check Volatility Environment",
                "step3": "Confirm Strength & Momentum",
                "step4": "Identify Support & Resistance",
                "step5": "Select Optimal Strategy"
            }
        }
        
        logger.info(f"Options analysis completed successfully for {ticker}")
        return options_analysis
        
    except Exception as e:
        logger.error(f"Error in options analysis: {str(e)}")
        # Return a basic error response
        return {
            "ticker": ticker,
            "error": f"Unable to complete options analysis: {str(e)}",
            "strategy_recommendation": {
                "name": "Unable to determine",
                "rationale": "Analysis error - insufficient data"
            }
        }

def get_options_cheatsheet_markdown() -> str:
    """Returns the options strategy cheatsheet as formatted markdown for display"""
    return get_analyst_cheatsheet_markdown()
