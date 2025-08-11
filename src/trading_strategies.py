# =========================
# Imports
# =========================
import pandas as pd
# import json  # Uncomment if using JSON output

# =========================
# Advanced Options Trading Strategies Data
# Professional-Level Trading Guide for Hedge Fund & Institutional Traders
# =========================
strategies_data = [
    {
        "Strategy": "Day Trading Calls/Puts",
        "Description": "Intraday buying/selling of calls or puts to exploit fast price swings, closing before market close. Professional-grade scalping requires ultra-fast execution and real-time market data access.",
        "Timeframe": "Minutes to hours (intraday). Positions are closed before the market closes.",
        "Key_Indicators": [
            "1-minute RSI & Stochastic: Identify overbought/oversold momentum swings",
            "VWAP: Intraday support/resistance for entries/exits", 
            "Level 2 & Order Flow: Spot large institutional buyers/sellers",
            "Bollinger Bands (1-min): Volatility squeeze and breakout signals",
            "Volume spikes confirming momentum shifts"
        ],
        "When_To_Use": [
            "High volatility catalysts (earnings, Fed announcements, geopolitical events)",
            "Strong momentum stocks or ETFs showing clear directional bias intraday",
            "When you have ultra-fast execution and real-time Level 2 data access",
            "Market open/close volatility plays with proper risk management"
        ],
        "Advanced_Tips": [
            "Trade only highly liquid options with tight bid-ask spreads",
            "Monitor open interest and volume for liquidity confirmation",
            "Use trailing stops or market orders triggered by volume spikes",
            "Avoid trading at market open/close unless very experienced with volatility",
            "Implement strict daily loss limits and position sizing rules"
        ],
        "Risk_Management": [
            "Use defined risk per trade (typically 1-2% of account)",
            "Monitor gamma risk and delta exposure continuously",
            "Have pre-defined exit strategies for both profit and loss scenarios",
            "Account for slippage and commission costs in profit calculations"
        ],
        "Pros": [
            "High potential for quick profits if the underlying asset moves significantly in the anticipated direction within the day.",
            "Limited overnight risk: Positions are closed before the market closes, avoiding overnight price gaps.",
            "Leverage: Options amplify price movements, allowing for potentially larger gains with a smaller capital outlay.",
            "Multiple opportunities throughout trading session"
        ],
        "Cons": [
            "Extremely high risk: Price movements can be unpredictable, and losses can be substantial.",
            "Requires constant monitoring: Demands intense focus and quick decision-making skills.",
            "High transaction costs: Frequent trading leads to higher commission and fee expenses.",
            "Time Decay (Theta): Options lose value as they approach expiration, and this effect is amplified closer to the expiration date.",
            "Volatility Risk: Implied volatility can change rapidly, impacting option prices.",
            "Execution risk and slippage in fast markets"
        ],
        "When to Use": "Strong conviction about the direction of an asset's price movement within a single trading day; high tolerance for risk and ability to monitor the market continuously; well-defined trading plan with strict entry and exit rules.",
        "Timeframe Used With": "N/A - This is a timeframe-based strategy.",
        "Suitable For": "Experienced traders with a deep understanding of technical analysis, market dynamics, and risk management. Requires professional-grade execution infrastructure."
    },
    {
        "Strategy": "Iron Condor",
        "Description": "A neutral strategy that profits when the underlying asset's price remains within a defined range. It involves selling an out-of-the-money (OTM) call spread and an OTM put spread simultaneously. Professional implementation focuses on high-probability trades with optimal IV conditions.",
        "Timeframe": "Typically held for a few weeks to a month, aiming to profit from time decay.",
        "Key_Indicators": [
            "IV Rank & IV Percentile: Ensure selling premium at inflated IV (above 50th percentile)",
            "ADX below 20: Confirms low trend strength and range-bound price action",
            "Delta Neutral Positioning: Sum delta near zero for market neutrality",
            "Historical volatility vs implied volatility analysis",
            "Support and resistance levels for strike selection"
        ],
        "When_To_Use": [
            "Markets exhibiting low volatility or range-bound behavior",
            "When IV Rank is elevated (sell premium when options are expensive)",
            "Ahead of earnings or events expected to produce minimal price moves",
            "During market consolidation phases with clear support/resistance"
        ],
        "Advanced_Tips": [
            "Adjust strikes dynamically based on risk tolerance and margin requirements",
            "Monitor position daily; roll wings or close if price moves too close to short strikes",
            "Use defined risk software to calculate max loss and margin requirements",
            "Consider iron condor variations (iron butterfly, jade lizard) for different market conditions",
            "Scale into positions during high IV environments"
        ],
        "Risk_Management": [
            "Set profit targets at 25-50% of maximum profit potential",
            "Have adjustment strategies ready (rolling, closing wings)",
            "Monitor underlying technical levels continuously",
            "Use position sizing appropriate for portfolio volatility"
        ],
        "Pros": [
            "Limited risk: The maximum loss is capped and known in advance.",
            "High probability of profit: Profitable if the price stays within the defined range.",
            "Benefits from time decay: As time passes, the options lose value, increasing the profit potential.",
            "Generates consistent income in range-bound markets"
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Requires adjustments: If the price moves outside the defined range, adjustments may be necessary, which can be complex and costly.",
            "Commissions: Four legs of the trade mean higher commission costs.",
            "Early Assignment Risk: While rare, there's a risk of early assignment, especially close to expiration.",
            "Large losses possible if underlying breaks out of range"
        ],
        "When to Use": "Expect the underlying asset's price to remain relatively stable; implied volatility is high, increasing the premium received for selling the options.",
        "Timeframe Used With": "Typically used with options expiring in 30-60 days.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with range-bound trading and have sophisticated risk management systems."
    },
    {
        "Strategy": "Straddle/Strangle",
        "Description": "Straddle: Buying a call and a put option with the same strike price and expiration date. Profitable if the price moves significantly in either direction. Strangle: Buying a call and a put option with different strike prices (OTM) but the same expiration date. Requires a larger price movement than a straddle to become profitable. Professional implementation focuses on volatility timing and catalyst-driven events.",
        "Timeframe": "Short-term, typically held for a few days to a few weeks.",
        "Key_Indicators": [
            "IV Rank low (below 30): Buy cheap volatility before major events",
            "Historical volatility vs implied volatility: Confirm IV is underpriced relative to realized moves",
            "Volume spike & news flow: Confirm catalyst proximity and market attention",
            "Gamma exposure and acceleration potential",
            "Put-call ratio for sentiment analysis"
        ],
        "When_To_Use": [
            "Before major, unpredictable catalysts (earnings, FDA approvals, geopolitical events)",
            "When IV is low to avoid paying rich premiums",
            "When expecting high volatility spikes but unsure about direction",
            "Around binary events with unclear outcomes"
        ],
        "Advanced_Tips": [
            "Calculate breakeven points carefully; price move must exceed combined premium paid",
            "Consider scaling out if large moves occur before expiration to lock profits",
            "Use short-dated options for quicker decay but higher gamma exposure",
            "Monitor volatility skew for optimal strike selection",
            "Consider ratio adjustments if directional bias develops"
        ],
        "Risk_Management": [
            "Set profit targets at 50-100% gains on volatility expansion",
            "Have exit strategies for time decay acceleration",
            "Monitor delta exposure and hedge if needed",
            "Use position sizing appropriate for volatility risk"
        ],
        "Pros": [
            "Unlimited profit potential if the price moves significantly in either direction.",
            "Profits from volatility: Benefits from an increase in implied volatility.",
            "Directionally neutral: Doesn't require predicting the direction of the price movement, only the magnitude.",
            "Excellent for binary event trading"
        ],
        "Cons": [
            "Limited time: The price must move significantly before expiration to be profitable.",
            "Time decay: Both the call and put options lose value as time passes.",
            "Requires a large price movement: The price must move beyond the breakeven points to generate a profit.",
            "Expensive if purchased when IV is elevated"
        ],
        "When to Use": "Expect a significant price movement in the underlying asset but are unsure of the direction; before a major news announcement or event likely to cause price volatility.",
        "Timeframe Used With": "Typically used with options expiring in 1-4 weeks.",
        "Suitable For": "Traders with a moderate to high risk tolerance who are comfortable with volatile markets and have sophisticated volatility analysis capabilities."
    },
    {
        "Strategy": "Butterfly Spread",
        "Description": "A limited risk, limited profit strategy designed to profit when the underlying asset's price remains near a specific strike price. It involves buying two options at different strike prices and selling two options at a strike price in between. Can be created with calls or puts. Professional implementation focuses on precise technical level targeting.",
        "Timeframe": "Typically held for a few weeks to a month.",
        "Key_Indicators": [
            "Strong support/resistance levels identified via pivot points, Fibonacci, or VWAP",
            "Low IV environments (IV Rank < 40) for cost efficiency",
            "Price consolidations with low ADX (trending strength below 25)",
            "Volume profile showing significant trading activity at target price level",
            "Technical confluence zones with multiple indicators"
        ],
        "When_To_Use": [
            "Expecting low volatility/stable price action around a strong technical level",
            "When IV is low to medium, as volatility increase hurts profitability",
            "To reduce cost versus straddles or strangles when confident about price zone",
            "During earnings season for stocks with predictable post-earnings ranges"
        ],
        "Advanced_Tips": [
            "Use liquid options to avoid bid-ask slippage on multi-leg orders",
            "Monitor theta decay—profit maximizes as expiration nears if price stays near middle strike",
            "Plan exit strategy if price breaks out of expected range",
            "Consider iron butterfly variations for credit spreads",
            "Use technical analysis to select optimal center strike price"
        ],
        "Risk_Management": [
            "Set profit targets at 25-50% of maximum profit",
            "Have breakout protection strategies ready",
            "Monitor position delta and gamma exposure",
            "Use appropriate position sizing for limited profit potential"
        ],
        "Pros": [
            "Limited risk: The maximum loss is capped and known in advance.",
            "High probability of profit: Profitable if the price stays near the middle strike price.",
            "Lower cost than a straddle/strangle: Requires less capital than buying a straddle or strangle.",
            "Excellent risk-reward ratio when properly timed"
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Complex to manage: Requires careful selection of strike prices and monitoring of the position.",
            "Low liquidity: Can be difficult to enter and exit the position if the options are not actively traded.",
            "Sensitive to volatility changes"
        ],
        "When to Use": "Expect the underlying asset's price to remain relatively stable and near a specific strike price; want to profit from time decay while limiting risk.",
        "Timeframe Used With": "Typically used with options expiring in 30-60 days.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with range-bound trading and have a specific price target in mind, backed by strong technical analysis."
    },
    {
        "Strategy": "Scalping Options",
        "Description": "A high-frequency trading strategy that involves making small profits from small price movements in options contracts. Traders typically hold positions for only a few seconds to a few minutes. Professional implementation requires institutional-grade infrastructure and sophisticated risk controls.",
        "Timeframe": "Seconds to minutes (intraday).",
        "Key_Indicators": [
            "Order book imbalance and fast Level 2 data analysis",
            "Tick charts and very short time frame momentum oscillators",
            "Market microstructure signals and order flow analysis",
            "Real-time bid-ask spread monitoring",
            "Volume-weighted average price (VWAP) deviations"
        ],
        "When_To_Use": [
            "In highly liquid markets (index options like SPX, SPY, QQQ)",
            "During high volume spikes and low latency execution environments", 
            "When using algorithmic or automated trading for consistent execution",
            "Around market open/close and major news releases"
        ],
        "Advanced_Tips": [
            "Use DMA (Direct Market Access) and colocated servers if possible",
            "Keep commissions/fees very low to maintain profitability on small moves",
            "Implement strict daily loss limits to avoid catastrophic losses",
            "Use sophisticated risk management algorithms",
            "Focus on highly liquid strikes with tight bid-ask spreads"
        ],
        "Risk_Management": [
            "Implement real-time risk controls and position limits",
            "Use automated stop-loss systems",
            "Monitor exposure across all positions continuously",
            "Have kill switches ready for emergency position closure"
        ],
        "Pros": [
            "Potential for high profits: If executed correctly, scalping can generate significant profits over time.",
            "Limited risk per trade: Positions are held for very short periods, reducing the risk of large single losses.",
            "High frequency of trades: Allows for multiple opportunities to profit throughout the day.",
            "Can exploit market inefficiencies quickly"
        ],
        "Cons": [
            "Extremely high risk: Requires lightning-fast decision-making skills and the ability to react quickly to market changes.",
            "High transaction costs: Frequent trading leads to very high commission and fee expenses.",
            "Requires specialized tools and software: Demands access to real-time market data and advanced trading platforms.",
            "Stressful: The fast-paced nature of scalping can be mentally and emotionally demanding.",
            "Technology dependency and execution risk"
        ],
        "When to Use": "Deep understanding of market microstructure and order flow; access to high-speed internet and advanced trading tools; ability to remain calm and focused under pressure.",
        "Timeframe Used With": "N/A - This is a timeframe-based strategy.",
        "Suitable For": "Highly experienced and disciplined traders with a strong understanding of market dynamics and risk management. Requires professional-grade infrastructure. Most retail traders should avoid this strategy."
    },
    {
        "Strategy": "Swing Trading",
        "Description": "A short- to medium-term trading strategy that involves holding options positions for a few days to a few weeks. The goal is to profit from price swings in the underlying asset. Professional implementation combines technical analysis with fundamental catalysts for optimal timing.",
        "Timeframe": "Days to weeks.",
        "Key_Indicators": [
            "Daily MACD crossovers and 14-day RSI for trend confirmation",
            "Volume spikes confirming momentum and institutional interest",
            "Moving averages (20, 50 SMA/EMA) as dynamic support/resistance levels",
            "Fibonacci retracement levels for entry timing",
            "Sector rotation indicators and relative strength analysis"
        ],
        "When_To_Use": [
            "Clear trend development confirmed by technical and fundamental analysis",
            "Sector rotations or momentum shifts in the market",
            "Moderate risk tolerance with time for active position management",
            "When earnings or catalyst events align with technical setups"
        ],
        "Advanced_Tips": [
            "Use vertical spreads to mitigate time decay and reduce capital requirements",
            "Always set stop losses for overnight and gap risk management",
            "Align positions with fundamental catalysts (earnings, guidance changes, sector news)",
            "Consider rolling positions to extend time horizon if thesis remains intact",
            "Use position sizing based on volatility and time to expiration"
        ],
        "Risk_Management": [
            "Implement strict stop-loss levels based on technical support/resistance",
            "Monitor overnight gap risk and adjust position sizing accordingly",
            "Use protective strategies for event risk",
            "Track theta decay and adjust positions as needed"
        ],
        "Pros": [
            "Potential for higher profits than day trading: Allows for larger price movements to unfold.",
            "Less time commitment than day trading: Doesn't require constant monitoring of the market.",
            "Can be combined with other strategies: Can be used to hedge existing positions or generate income.",
            "Excellent risk-reward ratios when properly timed"
        ],
        "Cons": [
            "Exposure to overnight risk: Positions are held overnight, exposing them to potential price gaps.",
            "Requires patience: Can take time for price movements to unfold.",
            "Time decay: Options lose value as they approach expiration.",
            "Event risk from unexpected news or earnings"
        ],
        "When to Use": "Identify a clear trend in the underlying asset's price; moderate risk tolerance and comfortable holding positions overnight; well-defined trading plan with clear entry and exit rules.",
        "Timeframe Used With": "Typically used with options expiring in 2-8 weeks.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with short- to medium-term trading and have strong technical analysis skills."
    },
    {
        "Strategy": "Covered Calls",
        "Description": "Selling call options on shares of stock that you already own. The goal is to generate income from the premium received for selling the options. Professional implementation focuses on optimizing strike selection and timing for maximum income generation.",
        "Timeframe": "Typically held for a few weeks to a few months.",
        "Key_Indicators": [
            "Technical resistance levels to select strike prices just above recent highs",
            "Implied volatility analysis of calls to ensure premiums justify assignment risk",
            "Dividend calendar and ex-dividend date considerations",
            "Earnings calendar to avoid unwanted assignment around events",
            "Stock's historical trading range and volatility patterns"
        ],
        "When_To_Use": [
            "Moderately bullish to neutral outlook on stock holdings",
            "Holding stable, dividend-paying stocks with manageable volatility",
            "Desire to generate income while maintaining long equity exposure",
            "During periods of elevated implied volatility"
        ],
        "Advanced_Tips": [
            "Roll calls (up and out) if stock rallies aggressively toward strike",
            "Avoid covered calls on highly volatile or rapidly rising stocks",
            "Use technical analysis to time entries during resistance levels",
            "Consider using on portfolio holdings to enhance overall yield",
            "Monitor early assignment risk, especially near ex-dividend dates"
        ],
        "Risk_Management": [
            "Set strike prices above key resistance levels",
            "Monitor assignment probability and early assignment risk",
            "Have rolling strategies ready for unexpected moves",
            "Consider protective puts if downside protection is needed"
        ],
        "Pros": [
            "Generates income: Provides a steady stream of income from the premium received.",
            "Reduces portfolio volatility: Can help to offset losses in the underlying stock.",
            "Limited risk: The maximum loss is capped at the cost of the stock.",
            "Tax-efficient income generation strategy"
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped at the strike price of the call option.",
            "Opportunity cost: If the stock price rises above the strike price, you will be forced to sell your shares at the strike price, missing out on potential gains.",
            "Requires owning the underlying stock: Requires a significant capital outlay to purchase the shares.",
            "Assignment risk limits upside participation"
        ],
        "When to Use": "Neutral to bullish on the underlying stock and want to generate income from holdings; want to reduce the volatility of your portfolio.",
        "Timeframe Used With": "Typically used with options expiring in 1-3 months.",
        "Suitable For": "Investors who own shares of stock and want to generate income from their holdings while accepting limited upside potential."
    },
    {
        "Strategy": "Protective Puts",
        "Description": "Buying put options on shares of stock that you already own. The goal is to protect your portfolio from potential losses if the stock price declines. Professional implementation focuses on cost-effective downside protection and portfolio insurance strategies.",
        "Timeframe": "Typically held for a few weeks to a few months.",
        "Key_Indicators": [
            "Put-call parity analysis for fair pricing evaluation",
            "Implied volatility spikes which raise premium costs",
            "Technical support levels to assess potential downside risk",
            "VIX levels and market fear indicators",
            "Correlation analysis with broader market indices"
        ],
        "When_To_Use": [
            "Prior to earnings, market corrections, or increased uncertainty",
            "When downside protection cost is justified by risk assessment",
            "Managing tail risk or implementing portfolio insurance",
            "During periods of market instability or geopolitical tension"
        ],
        "Advanced_Tips": [
            "Consider put spreads (buy put, sell lower strike put) to reduce insurance cost",
            "Time protection purchases to match specific risk horizons (1–3 months typical)",
            "Avoid buying puts when IV is extremely high unless protection is critical",
            "Use collar strategies (protective put + covered call) for cost reduction",
            "Monitor delta and adjust hedge ratios based on portfolio exposure"
        ],
        "Risk_Management": [
            "Calculate optimal hedge ratios based on portfolio beta",
            "Monitor cost of protection relative to potential losses",
            "Set criteria for rolling or closing protection",
            "Consider dynamic hedging strategies for large portfolios"
        ],
        "Pros": [
            "Protects against losses: Limits the potential downside risk in your portfolio.",
            "Allows you to participate in potential gains: If the stock price rises, you can still profit from the upside.",
            "Simple to implement: Relatively easy to understand and execute.",
            "Peace of mind during volatile periods"
        ],
        "Cons": [
            "Costs money: The premium paid for the put options reduces your overall profit potential.",
            "Time decay: Put options lose value as they approach expiration.",
            "Requires owning the underlying stock: Requires a significant capital outlay to purchase the shares.",
            "Insurance cost can be significant during high volatility periods"
        ],
        "When to Use": "Concerned about a potential decline in the stock price and want to protect your portfolio; want to limit downside risk while still participating in potential gains.",
        "Timeframe Used With": "Typically used with options expiring in 1-3 months.",
        "Suitable For": "Investors who own shares of stock and want to protect their portfolios from potential losses, particularly during uncertain market conditions."
    },
    {
        "Strategy": "Vertical Spreads",
        "Description": "Buying and selling options of the same type (calls or puts) with the same expiration date but different strike prices. Can be bullish (debit call spread, credit put spread) or bearish (debit put spread, credit call spread). Professional implementation focuses on optimal strike selection and volatility timing.",
        "Timeframe": "Typically held for a few weeks to a few months.",
        "Key_Indicators": [
            "Implied volatility analysis to choose between debit or credit spreads",
            "Spread delta and probability ITM calculations for strike selection",
            "Trend confirmation with MACD, RSI, or price action signals",
            "Volume analysis to ensure liquidity at chosen strikes",
            "Risk-reward ratio optimization based on market conditions"
        ],
        "When_To_Use": [
            "Moderate directional views with defined risk and capital limits",
            "When you want clearly defined maximum loss and profit parameters",
            "To reduce option cost and minimize time decay impact",
            "During trending markets with clear directional bias"
        ],
        "Advanced_Tips": [
            "Use tight strike spacing in high volatility markets; wider spacing in low volatility",
            "Monitor IV changes—credit spreads benefit from falling IV, debit spreads from rising IV",
            "Have exit and rolling strategies planned before entry",
            "Consider width of spread based on underlying's typical trading range",
            "Use probability analysis to optimize strike selection"
        ],
        "Risk_Management": [
            "Set profit targets at 25-50% of maximum profit for credit spreads",
            "Monitor early assignment risk on short options",
            "Have rolling strategies ready for adverse moves",
            "Use appropriate position sizing based on defined risk"
        ],
        "Pros": [
            "Limited risk: The maximum loss is capped and known in advance.",
            "Lower cost than buying options outright: Requires less capital than buying a single call or put option.",
            "Can be tailored to different market conditions: Can be used in bullish, bearish, or neutral market environments.",
            "Excellent risk-reward profiles when properly structured"
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Requires careful selection of strike prices: The choice of strike prices can significantly impact the profitability of the trade.",
            "Assignment risk on short options",
            "Complexity in managing multi-leg positions"
        ],
        "When to Use": "Directional bias on the underlying asset and want to limit risk; want to reduce the cost of buying options outright.",
        "Timeframe Used With": "Typically used with options expiring in 30-60 days.",
        "Suitable For": "Traders with a moderate risk tolerance who have a directional bias on the underlying asset and understand multi-leg option strategies."
    },
    {
        "Strategy": "Calendar Spreads",
        "Description": "Buying and selling options of the same type (calls or puts) with the same strike price but different expiration dates. Typically involves selling a near-term option and buying a longer-term option. Professional implementation focuses on volatility term structure analysis and time decay optimization.",
        "Timeframe": "Typically held for a few weeks to a few months. The near-term option might expire in a few weeks, while the longer-term option expires a month or two later.",
        "Key_Indicators": [
            "IV term structure analysis for backwardation or contango opportunities",
            "Theta decay estimates comparing front month vs back month options",
            "Technical indicators showing consolidation or range-bound behavior",
            "Volatility skew analysis for optimal strike selection",
            "Earnings calendar to avoid unwanted volatility spikes"
        ],
        "When_To_Use": [
            "Expecting underlying price to remain relatively stable or move slightly in specific direction",
            "When front-month implied volatility is elevated relative to back-month",
            "To profit from time decay differential while limiting risk",
            "During low volatility environments with expectations of stability"
        ],
        "Advanced_Tips": [
            "Manage risk by rolling near-term options as expiration approaches",
            "Avoid calendar spreads if expecting sudden large moves or volatility spikes",
            "Use liquid, high-volume strikes to minimize bid-ask impact",
            "Monitor vega exposure and volatility term structure changes",
            "Consider ratio calendars for additional income generation"
        ],
        "Risk_Management": [
            "Have rolling strategies ready for near-term option management",
            "Monitor volatility changes that can impact spread value",
            "Set profit targets based on time decay acceleration",
            "Avoid holding through major events or earnings"
        ],
        "Pros": [
            "Profits from time decay: The near-term option loses value faster than the longer-term option, generating a profit.",
            "Can be used in neutral or slightly bullish/bearish market conditions: Profitable if the price stays near the strike price or moves slightly in the desired direction.",
            "Lower cost than buying options outright: Requires less capital than buying a single call or put option.",
            "Benefits from volatility term structure inefficiencies"
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Complex to manage: Requires careful monitoring of the position and potential adjustments as the expiration dates approach.",
            "Volatility Risk: Changes in implied volatility can significantly impact the profitability of the trade.",
            "Requires active management and sophisticated understanding"
        ],
        "When to Use": "Expect the underlying asset's price to remain relatively stable or move slightly in a specific direction; want to profit from time decay while limiting risk.",
        "Timeframe Used With": "Near-term options expiring in 2-4 weeks, longer-term options expiring in 1-3 months.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with range-bound trading and have a sophisticated understanding of time decay, volatility, and complex option strategies."
    }
]

# =========================
# Additional Strategy Analysis Functions
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
        'trending_up': ['Swing Trading', 'Vertical Spreads', 'Covered Calls'],
        'trending_down': ['Vertical Spreads', 'Protective Puts', 'Swing Trading'],
        'sideways': ['Iron Condor', 'Butterfly Spread', 'Calendar Spreads', 'Covered Calls'],
        'high_volatility': ['Straddle/Strangle', 'Iron Condor', 'Vertical Spreads'],
        'low_volatility': ['Calendar Spreads', 'Butterfly Spread', 'Covered Calls']
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
        safe_strategies = ['Covered Calls', 'Protective Puts', 'Calendar Spreads']
    elif risk_level == 'moderate':
        safe_strategies = ['Iron Condor', 'Butterfly Spread', 'Vertical Spreads', 'Swing Trading']
    else:  # aggressive
        safe_strategies = ['Day Trading Calls/Puts', 'Scalping Options', 'Straddle/Strangle']
    
    return [strategy for strategy in strategies_data if strategy['Strategy'] in safe_strategies]

def get_strategy_by_timeframe(timeframe):
    """
    Returns strategies suitable for specific trading timeframes.
    
    Args:
        timeframe (str): 'intraday', 'short_term', 'medium_term'
    
    Returns:
        list: Strategies matching the timeframe
    """
    if timeframe == 'intraday':
        suitable_strategies = ['Day Trading Calls/Puts', 'Scalping Options']
    elif timeframe == 'short_term':
        suitable_strategies = ['Swing Trading', 'Straddle/Strangle', 'Vertical Spreads']
    else:  # medium_term
        suitable_strategies = ['Covered Calls', 'Protective Puts', 'Calendar Spreads', 'Iron Condor']
    
    return [strategy for strategy in strategies_data if strategy['Strategy'] in suitable_strategies]

# =========================
# DataFrame Creation
# =========================
df = pd.DataFrame(strategies_data)
# print(df)

# =========================
# Optional: JSON Output
# =========================
# import json
# json_data = json.dumps(strategies_data, indent=4)
# print(json_data)
