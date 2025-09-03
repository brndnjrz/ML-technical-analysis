# =========================
# Imports
# =========================
import pandas as pd
import json

# =========================
# Advanced Options Trading Strategies Data
# Professional-Level Trading Guide for Hedge Fund & Institutional Traders
# =========================
strategies_data = [
    {
        "Strategy": "Covered Calls",
        "Description": "Sell call options against stock you own to generate income. Best used in neutral to slightly bullish markets.",
        "Timeframe": "Short-Term (1–7 days)",
        "Pros": ["Generates steady income", "Reduces downside risk", "Works well in flat or slightly bullish markets"],
        "Cons": ["Limits upside potential", "Requires owning 100 shares", "Assignment risk if stock rises sharply"],
        "When to Use": "Use when expecting neutral to slightly bullish price action and want to generate income from owned shares.",
        "Suitable For": "Income-focused investors with neutral to slightly bullish outlook who own shares.",
        "Timeframes": {
            "Short-Term (1–7 days)": {
                "Key_Indicators": [
                    "15m, 1h charts for technical resistance or pivot levels",
                    "Intraday VWAP to gauge mean reversion",
                    "ATR-based expected move for strike distance",
                    "IVR (Implied Volatility Rank) for premium quality",
                    "Volume profile to identify short-term supply zones"
                ],
                "Best_Use": "Capture rapid option decay while avoiding price spikes near resistance",
                "Advanced_Tips": [
                    "Focus on high-liquidity stocks to minimize slippage",
                    "Align strike selection with short-term support/resistance and IV spikes",
                    "Monitor option Greeks (Delta/Gamma) to manage risk",
                    "Consider rolling options if underlying moves against position",
                    "Check for upcoming earnings/events that could spike IV"
                ]
            },
            "Medium/Long-Term (2–12 weeks)": {
                "Key_Indicators": [
                    "Daily/weekly charts for major resistance",
                    "Trend strength via moving averages (50/200 SMA/EMA)",
                    "IVR and historical volatility to assess premium potential",
                    "Earnings and event calendar to avoid surprise moves"
                ],
                "Best_Use": "Generate steady premium income and hedge against moderate price movement",
                "Advanced_Tips": [
                    "Select strikes just out-of-the-money for optimal premium vs risk",
                    "Use longer-term trend lines to time entries",
                    "Analyze sector rotation for potential upside/volatility changes",
                    "Combine with stock dividends to increase total return",
                    "Regularly monitor IV rank changes to adjust positions"
                ]
            }
        }
    },
    {
        "Strategy": "Cash-Secured Puts",
        "Description": "Sell put options with enough cash to buy the stock if assigned. Used to acquire stock at a discount or generate income.",
        "Timeframe": "Short-Term (1–7 days)",
        "Pros": ["Acquire stock at a discount", "Earn premium income", "Defined risk (cash reserved)"],
        "Cons": ["Requires large cash reserve", "Assignment risk in falling markets", "Limited upside if stock rallies"],
        "When to Use": "Use when willing to buy stock at a lower price or to generate income in sideways to slightly bullish markets.",
        "Suitable For": "Investors willing to buy stock at a discount or generate income.",
        "Timeframes": {
            "Short-Term (1–7 days)": {
                "Key_Indicators": [
                    "15m, 1h charts for intraday support",
                    "Intraday VWAP and anchored VWAP",
                    "ATR for short-term expected move",
                    "RSI/Stochastic on lower timeframes for oversold entries",
                    "Options flow/order flow for bullish pressure"
                ],
                "Best_Use": "Quick premium capture with potential stock entry at support levels",
                "Advanced_Tips": [
                    "Target options near key intraday support levels",
                    "Monitor volume and open interest to gauge liquidity",
                    "Use delta to select strikes with higher probability of expiring OTM",
                    "Exit positions early if support breaks or momentum shifts",
                    "Avoid selling puts during high-impact news events"
                ]
            },
            "Medium/Long-Term (30–60 days)": {
                "Key_Indicators": [
                    "Daily/weekly support levels",
                    "Trend confirmation via moving averages",
                    "ATR for position sizing and risk",
                    "IVR for premium quality",
                    "Upcoming earnings or corporate events"
                ],
                "Best_Use": "Earn premium or acquire stock at discount while maintaining capital efficiency",
                "Advanced_Tips": [
                    "Place strikes slightly below strong support for lower assignment risk",
                    "Use IVR to maximize premium collection in high-volatility periods",
                    "Diversify across uncorrelated sectors to reduce risk",
                    "Consider rolling short puts if stock moves significantly",
                    "Track open interest shifts to anticipate institutional activity"
                ]
            }
        }
    },
    {
        "Strategy": "Iron Condors",
        "Description": "Sell both a call spread and a put spread to profit from low volatility and range-bound price action.",
        "Timeframe": "Short-Term (1–7 days)",
        "Pros": ["Profits from time decay", "Defined risk", "Works best in low volatility"],
        "Cons": ["Losses if price breaks out of range", "Requires careful strike selection", "Multiple legs increase commissions"],
        "When to Use": "Use when expecting low volatility and range-bound price action.",
        "Suitable For": "Traders expecting low volatility and range-bound price action.",
        "Timeframes": {
            "Short-Term (1–7 days)": {
                "Key_Indicators": [
                    "15m, 1h charts for near-term support/resistance",
                    "ATR relative to strike distance",
                    "Implied move from options pricing to set wings",
                    "IVR to ensure rich premiums",
                    "Bollinger Bands for range confirmation"
                ],
                "Best_Use": "Profit from time decay in tightly ranged markets over weekly options",
                "Advanced_Tips": [
                    "Use weekly options with high IV for maximum theta decay",
                    "Adjust wings if underlying moves close to short strikes",
                    "Monitor volume spikes to anticipate breakouts",
                    "Avoid entering just before major economic events",
                    "Use delta-neutral placement to reduce directional risk"
                ]
            },
            "Medium/Long-Term (30–45 days)": {
                "Key_Indicators": [
                    "Daily/weekly support/resistance zones",
                    "ATR for expected range",
                    "IV percentile to select optimal premiums",
                    "Trend neutrality confirmation",
                    "Low momentum readings to avoid breakout risk"
                ],
                "Best_Use": "Collect premium in range-bound markets with defined risk over longer expirations",
                "Advanced_Tips": [
                    "Select strikes outside expected move based on historical volatility",
                    "Roll spreads as expiration approaches if underlying nears short strikes",
                    "Combine with market-neutral hedges for added protection",
                    "Monitor IV rank changes to adjust timing or strikes",
                    "Consider correlation with sector ETFs for additional risk assessment"
                ]
            }
        }
    },
    {
        "Strategy": "Credit Spreads",
        "Description": "Sell one option and buy another further out-of-the-money to limit risk. Used for directional trades with defined risk.",
        "Timeframe": "Short-Term (1–7 days)",
        "Pros": ["Defined risk and reward", "Profits from time decay", "Flexible for bullish or bearish setups"],
        "Cons": ["Limited profit potential", "Losses if price moves against position", "Requires margin approval"],
        "When to Use": "Use for defined-risk directional trades in moderately trending markets.",
        "Suitable For": "Directional traders seeking defined risk and reward.",
        "Timeframes": {
            "Short-Term (1–7 days)": {
                "Key_Indicators": [
                    "15m, 1h charts for short-term support (puts) or resistance (calls)",
                    "VWAP and intraday moving averages for bias confirmation",
                    "Delta 0.20–0.30 for short strike selection",
                    "IVR for premium quality",
                    "ATR to ensure strikes beyond expected move",
                    "RSI/MACD for momentum confirmation on intraday charts"
                ],
                "Best_Use": "Capture high-probability directional premium with fast decay",
                "Advanced_Tips": [
                    "Focus on liquid options to reduce slippage",
                    "Place spreads at delta levels matching probability targets",
                    "Close early if IV collapses or trade moves favorably",
                    "Avoid over-leveraging in volatile conditions",
                    "Track large block trades or unusual options activity for clues"
                ]
            },
            "Medium/Long-Term (30–45 days)": {
                "Key_Indicators": [
                    "Daily/weekly support/resistance",
                    "Trend confirmation via moving averages",
                    "Delta 0.25–0.35 for strike selection",
                    "IVR for premium richness",
                    "ATR for positioning outside expected move",
                    "MACD/RSI on daily chart for directional bias"
                ],
                "Best_Use": "Directional defined-risk trades with higher confidence and lower gamma risk",
                "Advanced_Tips": [
                    "Use spreads around confirmed trend lines to minimize risk",
                    "Monitor IVR and adjust strikes if volatility changes significantly",
                    "Hedge with opposite side spreads if trend reverses",
                    "Stack expirations for consistent premium flow",
                    "Consider market correlation and macro events for additional context"
                ]
            }
        }
    },
    {
        "Strategy": "Swing Trading",
        "Timeframe": "Short-Term (1–7 days)",
        "Pros": ["Captures larger moves than day trading", "Less time required than long-term investing", "Can use options for leverage"],
        "Cons": ["Subject to overnight risk", "Requires monitoring for reversals", "May miss long-term trends"],
        "Timeframes": {
            "Short-Term (1–7 days)": {
                "Key_Indicators": [
                    "15m, 1h charts for early trend identification",
                    "VWAP for intraday bias",
                    "ATR for short-term volatility-based stops",
                    "RSI/Stochastic on intraday for overbought/oversold detection",
                    "Volume spikes confirming momentum shifts"
                ],
                "Best_Use": "Catch short-term swings aligned with multi-day trends",
                "Advanced_Tips": [
                    "Use intraday pullbacks for optimal entry points",
                    "Focus on high-volume breakouts for confirmation",
                    "Scale positions to reduce risk in volatile moves",
                    "Monitor correlation with sector indices",
                    "Combine with short-term option structures like vertical spreads"
                ]
            },
            "Medium/Long-Term (2–8 weeks)": {
                "Key_Indicators": [
                    "Daily/4-hour MACD crossovers for momentum",
                    "14-day RSI for overbought/oversold",
                    "ATR for position sizing and volatility-based stops",
                    "Fibonacci levels for target zones",
                    "Moving averages (20,50,200) for trend support/resistance",
                    "Sector strength vs. market index",
                    "IVR and percentile for option selection"
                ],
                "Best_Use": "Capture larger trend-based swings using options with moderate expirations",
                "Advanced_Tips": [
                    "Roll positions forward to extend trend capture",
                    "Use trailing stops to lock in gains while allowing upside",
                    "Hedge overnight exposure with inverse ETFs or offsetting options",
                    "Time entries around key catalysts like earnings or macro events",
                    "Adjust strikes using ATR and volatility forecasts to manage risk"
                ]
            }
        }
    },
    {
        "Strategy": "Day Trading Calls/Puts",
        "Description": "Buy calls or puts for intraday moves, closing positions before the end of the trading day to avoid overnight risk.",
        "Timeframe": "Intraday (minutes to hours)",
        "Pros": ["No overnight risk", "Quick profit/loss realization", "High number of trading opportunities"],
        "Cons": ["Requires constant attention", "High commissions/fees", "Can be stressful and risky"],
        "When to Use": "Use for intraday moves when you want to avoid overnight risk and capitalize on quick price swings.",
        "Suitable For": "Active traders seeking intraday opportunities and quick profits.",
        "Timeframes": {
            "Intraday (minutes to hours)": {
                "Key_Indicators": [
                    "1-min, 5-min charts for ultra-short momentum (RSI, Stochastic)",
                    "VWAP for intraday support/resistance",
                    "Level 2 & order flow to spot large buyers/sellers",
                    "Bollinger Bands for volatility squeeze/breakout",
                    "Volume spikes confirming momentum shifts"
                ],
                "Best_Use": "Exploit rapid price swings for intraday gains; requires real-time monitoring and execution",
                "Advanced_Tips": [
                    "Trade only highly liquid options to reduce slippage",
                    "Use real-time Level 2 data for order flow insights",
                    "Implement strict risk per trade (1–2% of account)",
                    "Avoid trading during low liquidity periods or news uncertainty",
                    "Use tight stops and quick exits to preserve capital"
                ]
            }
        }
    }
]

# =========================
# Strategy Analysis Functions
# =========================

def get_strategy_by_market_condition(market_condition):
    """
    Returns optimal strategies based on current market conditions.
    
    Args:
        market_condition (str): 'trending_up', 'trending_down', 'sideways', 'high_volatility', 'low_volatility'
    
    Returns:
        list: Recommended strategies for the given market condition
    """
    condition_mapping = {
        'trending_up': ['Covered Calls', 'Swing Trading'],
        'trending_down': ['Cash-Secured Puts', 'Credit Spreads'],
        'sideways': ['Iron Condors', 'Covered Calls'],
        'high_volatility': ['Day Trading Calls/Puts', 'Iron Condors'],
        'low_volatility': ['Covered Calls', 'Cash-Secured Puts']
    }
    
    recommended_strategies = condition_mapping.get(market_condition, [])
    return [strategy for strategy in strategies_data if strategy['Strategy'] in recommended_strategies]

def get_strategy_by_risk_tolerance(risk_level):
    """
    Returns strategies filtered by risk tolerance level.
    
    Args:
        risk_level (str): 'conservative', 'moderate', 'aggressive'
    
    Returns:
        list: Strategies suitable for the specified risk level
    """
    if risk_level == 'conservative':
        safe_strategies = ['Covered Calls', 'Cash-Secured Puts']
    elif risk_level == 'moderate':
        safe_strategies = ['Iron Condors', 'Credit Spreads', 'Swing Trading']
    else:  # aggressive
        safe_strategies = ['Day Trading Calls/Puts', 'Credit Spreads']
    
    return [strategy for strategy in strategies_data if strategy['Strategy'] in safe_strategies]

def get_strategy_by_timeframe(timeframe):
    """
    Returns strategies suitable for specific trading timeframes.
    
    Args:
        timeframe (str): 'short_term', 'medium_term', 'long_term'
    
    Returns:
        list: Strategies matching the timeframe
    """
    matching_strategies = []
    
    for strategy in strategies_data:
        timeframes = strategy.get('Timeframes', {})
        
        if timeframe == 'short_term':
            # Look for short-term timeframes in the strategy
            if any('Short-Term' in tf or 'Intraday' in tf for tf in timeframes.keys()):
                matching_strategies.append(strategy)
        elif timeframe == 'medium_term':
            if any('Medium' in tf or 'Long-Term' in tf for tf in timeframes.keys()):
                matching_strategies.append(strategy)
        else:  # long_term
            # Look for long-term timeframes
            if any('Long-Term' in tf or 'weeks' in tf for tf in timeframes.keys()):
                matching_strategies.append(strategy)
    
    return matching_strategies

def get_strategy_names():
    """
    Returns a list of all available strategy names.
    
    Returns:
        list: List of strategy names
    """
    return [strategy['Strategy'] for strategy in strategies_data]

def get_strategy_by_name(strategy_name):
    """
    Returns strategy data by name.
    
    Args:
        strategy_name (str): Name of the strategy
    
    Returns:
        dict: Strategy data or None if not found
    """
    for strategy in strategies_data:
        if strategy['Strategy'] == strategy_name:
            return strategy
    return None

def get_timeframes_for_strategy(strategy_name):
    """
    Returns available timeframes for a specific strategy.
    
    Args:
        strategy_name (str): Name of the strategy
    
    Returns:
        list: List of available timeframes for the strategy
    """
    strategy = get_strategy_by_name(strategy_name)
    if strategy and 'Timeframes' in strategy:
        return list(strategy['Timeframes'].keys())
    return []

# =========================
# DataFrame Creation
# =========================
def create_strategies_dataframe():
    """
    Creates a flattened DataFrame from strategies data for analysis.
    
    Returns:
        pd.DataFrame: Flattened strategies data
    """
    flattened_data = []
    
    for strategy in strategies_data:
        strategy_name = strategy['Strategy']
        timeframes = strategy.get('Timeframes', {})
        
        for timeframe, details in timeframes.items():
            flattened_data.append({
                'Strategy': strategy_name,
                'Timeframe': timeframe,
                'Best_Use': details.get('Best_Use', ''),
                'Key_Indicators_Count': len(details.get('Key_Indicators', [])),
                'Advanced_Tips_Count': len(details.get('Advanced_Tips', []))
            })
    
    return pd.DataFrame(flattened_data)

# Create DataFrame for backward compatibility
df = create_strategies_dataframe()

# =========================
# Optional: JSON Output
# =========================
def export_strategies_as_json():
    """
    Export strategies data as JSON string
    
    Returns:
        str: JSON formatted strategies data
    """
    return json.dumps(strategies_data, indent=4)

# For testing the module directly
if __name__ == "__main__":
    print("Available Strategies:")
    for name in get_strategy_names():
        print(f"- {name}")
    
    print("\nStrategy DataFrame:")
    print(df.head())
    
    print("\nExample Strategy (Covered Calls):")
    covered_calls = get_strategy_by_name("Covered Calls")
    if covered_calls:
        print(json.dumps(covered_calls, indent=2))
