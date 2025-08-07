# =========================
# Imports
# =========================
import pandas as pd
# import json  # Uncomment if using JSON output

# =========================
# Options Trading Strategies Data
# =========================
strategies_data = [
    {
        "Strategy": "Day Trading Calls/Puts",
        "Description": "Buying or selling call/put options with the intention of closing the position within the same trading day. The goal is to profit from short-term price fluctuations.",
        "Timeframe": "Minutes to hours (intraday). Positions are closed before the market closes.",
        "Pros": [
            "High potential for quick profits if the underlying asset moves significantly in the anticipated direction within the day.",
            "Limited overnight risk: Positions are closed before the market closes, avoiding overnight price gaps.",
            "Leverage: Options amplify price movements, allowing for potentially larger gains with a smaller capital outlay."
        ],
        "Cons": [
            "Extremely high risk: Price movements can be unpredictable, and losses can be substantial.",
            "Requires constant monitoring: Demands intense focus and quick decision-making skills.",
            "High transaction costs: Frequent trading leads to higher commission and fee expenses.",
            "Time Decay (Theta): Options lose value as they approach expiration, and this effect is amplified closer to the expiration date.",
            "Volatility Risk: Implied volatility can change rapidly, impacting option prices."
        ],
        "When to Use": "Strong conviction about the direction of an asset's price movement within a single trading day; high tolerance for risk and ability to monitor the market continuously; well-defined trading plan with strict entry and exit rules.",
        "Timeframe Used With": "N/A - This is a timeframe-based strategy.",
        "Suitable For": "Experienced traders with a deep understanding of technical analysis, market dynamics, and risk management."
    },
    {
        "Strategy": "Iron Condor",
        "Description": "A neutral strategy that profits when the underlying asset's price remains within a defined range. It involves selling an out-of-the-money (OTM) call spread and an OTM put spread simultaneously.",
        "Timeframe": "Typically held for a few weeks to a month, aiming to profit from time decay.",
        "Pros": [
            "Limited risk: The maximum loss is capped and known in advance.",
            "High probability of profit: Profitable if the price stays within the defined range.",
            "Benefits from time decay: As time passes, the options lose value, increasing the profit potential."
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Requires adjustments: If the price moves outside the defined range, adjustments may be necessary, which can be complex and costly.",
            "Commissions: Four legs of the trade mean higher commission costs.",
            "Early Assignment Risk: While rare, there's a risk of early assignment, especially close to expiration."
        ],
        "When to Use": "Expect the underlying asset's price to remain relatively stable; implied volatility is high, increasing the premium received for selling the options.",
        "Timeframe Used With": "Typically used with options expiring in 30-60 days.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with range-bound trading."
    },
    {
        "Strategy": "Straddle/Strangle",
        "Description": "Straddle: Buying a call and a put option with the same strike price and expiration date. Profitable if the price moves significantly in either direction. Strangle: Buying a call and a put option with different strike prices (OTM) but the same expiration date. Requires a larger price movement than a straddle to become profitable.",
        "Timeframe": "Short-term, typically held for a few days to a few weeks.",
        "Pros": [
            "Unlimited profit potential if the price moves significantly in either direction.",
            "Profits from volatility: Benefits from an increase in implied volatility.",
            "Directionally neutral: Doesn't require predicting the direction of the price movement, only the magnitude."
        ],
        "Cons": [
            "Limited time: The price must move significantly before expiration to be profitable.",
            "Time decay: Both the call and put options lose value as time passes.",
            "Requires a large price movement: The price must move beyond the breakeven points to generate a profit."
        ],
        "When to Use": "Expect a significant price movement in the underlying asset but are unsure of the direction; before a major news announcement or event likely to cause price volatility.",
        "Timeframe Used With": "Typically used with options expiring in 1-4 weeks.",
        "Suitable For": "Traders with a moderate to high risk tolerance who are comfortable with volatile markets."
    },
    {
        "Strategy": "Butterfly Spread",
        "Description": "A limited risk, limited profit strategy designed to profit when the underlying asset's price remains near a specific strike price. It involves buying two options at different strike prices and selling two options at a strike price in between. Can be created with calls or puts.",
        "Timeframe": "Typically held for a few weeks to a month.",
        "Pros": [
            "Limited risk: The maximum loss is capped and known in advance.",
            "High probability of profit: Profitable if the price stays near the middle strike price.",
            "Lower cost than a straddle/strangle: Requires less capital than buying a straddle or strangle."
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Complex to manage: Requires careful selection of strike prices and monitoring of the position.",
            "Low liquidity: Can be difficult to enter and exit the position if the options are not actively traded."
        ],
        "When to Use": "Expect the underlying asset's price to remain relatively stable and near a specific strike price; want to profit from time decay while limiting risk.",
        "Timeframe Used With": "Typically used with options expiring in 30-60 days.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with range-bound trading and have a specific price target in mind."
    },
    {
        "Strategy": "Scalping Options",
        "Description": "A high-frequency trading strategy that involves making small profits from small price movements in options contracts. Traders typically hold positions for only a few seconds to a few minutes.",
        "Timeframe": "Seconds to minutes (intraday).",
        "Pros": [
            "Potential for high profits: If executed correctly, scalping can generate significant profits over time.",
            "Limited risk: Positions are held for very short periods, reducing the risk of large losses.",
            "High frequency of trades: Allows for multiple opportunities to profit throughout the day."
        ],
        "Cons": [
            "Extremely high risk: Requires lightning-fast decision-making skills and the ability to react quickly to market changes.",
            "High transaction costs: Frequent trading leads to very high commission and fee expenses.",
            "Requires specialized tools and software: Demands access to real-time market data and advanced trading platforms.",
            "Stressful: The fast-paced nature of scalping can be mentally and emotionally demanding."
        ],
        "When to Use": "Deep understanding of market microstructure and order flow; access to high-speed internet and advanced trading tools; ability to remain calm and focused under pressure.",
        "Timeframe Used With": "N/A - This is a timeframe-based strategy.",
        "Suitable For": "Highly experienced and disciplined traders with a strong understanding of market dynamics and risk management. Most retail traders should avoid this."
    },
    {
        "Strategy": "Swing Trading",
        "Description": "A short- to medium-term trading strategy that involves holding options positions for a few days to a few weeks. The goal is to profit from price swings in the underlying asset.",
        "Timeframe": "Days to weeks.",
        "Pros": [
            "Potential for higher profits than day trading: Allows for larger price movements to unfold.",
            "Less time commitment than day trading: Doesn't require constant monitoring of the market.",
            "Can be combined with other strategies: Can be used to hedge existing positions or generate income."
        ],
        "Cons": [
            "Exposure to overnight risk: Positions are held overnight, exposing them to potential price gaps.",
            "Requires patience: Can take time for price movements to unfold.",
            "Time decay: Options lose value as they approach expiration."
        ],
        "When to Use": "Identify a clear trend in the underlying asset's price; moderate risk tolerance and comfortable holding positions overnight; well-defined trading plan with clear entry and exit rules.",
        "Timeframe Used With": "Typically used with options expiring in 2-8 weeks.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with short- to medium-term trading."
    },
    {
        "Strategy": "Covered Calls",
        "Description": "Selling call options on shares of stock that you already own. The goal is to generate income from the premium received for selling the options.",
        "Timeframe": "Typically held for a few weeks to a few months.",
        "Pros": [
            "Generates income: Provides a steady stream of income from the premium received.",
            "Reduces portfolio volatility: Can help to offset losses in the underlying stock.",
            "Limited risk: The maximum loss is capped at the cost of the stock."
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped at the strike price of the call option.",
            "Opportunity cost: If the stock price rises above the strike price, you will be forced to sell your shares at the strike price, missing out on potential gains.",
            "Requires owning the underlying stock: Requires a significant capital outlay to purchase the shares."
        ],
        "When to Use": "Neutral to bullish on the underlying stock and want to generate income from holdings; want to reduce the volatility of your portfolio.",
        "Timeframe Used With": "Typically used with options expiring in 1-3 months.",
        "Suitable For": "Investors who own shares of stock and want to generate income from their holdings."
    },
    {
        "Strategy": "Protective Puts",
        "Description": "Buying put options on shares of stock that you already own. The goal is to protect your portfolio from potential losses if the stock price declines.",
        "Timeframe": "Typically held for a few weeks to a few months.",
        "Pros": [
            "Protects against losses: Limits the potential downside risk in your portfolio.",
            "Allows you to participate in potential gains: If the stock price rises, you can still profit from the upside.",
            "Simple to implement: Relatively easy to understand and execute."
        ],
        "Cons": [
            "Costs money: The premium paid for the put options reduces your overall profit potential.",
            "Time decay: Put options lose value as they approach expiration.",
            "Requires owning the underlying stock: Requires a significant capital outlay to purchase the shares."
        ],
        "When to Use": "Concerned about a potential decline in the stock price and want to protect your portfolio; want to limit downside risk while still participating in potential gains.",
        "Timeframe Used With": "Typically used with options expiring in 1-3 months.",
        "Suitable For": "Investors who own shares of stock and want to protect their portfolios from potential losses."
    },
    {
        "Strategy": "Vertical Spreads",
        "Description": "Buying and selling options of the same type (calls or puts) with the same expiration date but different strike prices. Can be bullish (debit call spread, credit put spread) or bearish (debit put spread, credit call spread).",
        "Timeframe": "Typically held for a few weeks to a few months.",
        "Pros": [
            "Limited risk: The maximum loss is capped and known in advance.",
            "Lower cost than buying options outright: Requires less capital than buying a single call or put option.",
            "Can be tailored to different market conditions: Can be used in bullish, bearish, or neutral market environments."
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Requires careful selection of strike prices: The choice of strike prices can significantly impact the profitability of the trade."
        ],
        "When to Use": "Directional bias on the underlying asset and want to limit risk; want to reduce the cost of buying options outright.",
        "Timeframe Used With": "Typically used with options expiring in 30-60 days.",
        "Suitable For": "Traders with a moderate risk tolerance who have a directional bias on the underlying asset."
    },
    {
        "Strategy": "Calendar Spreads",
        "Description": "Buying and selling options of the same type (calls or puts) with the same strike price but different expiration dates. Typically involves selling a near-term option and buying a longer-term option.",
        "Timeframe": "Typically held for a few weeks to a few months. The near-term option might expire in a few weeks, while the longer-term option expires a month or two later.",
        "Pros": [
            "Profits from time decay: The near-term option loses value faster than the longer-term option, generating a profit.",
            "Can be used in neutral or slightly bullish/bearish market conditions: Profitable if the price stays near the strike price or moves slightly in the desired direction.",
            "Lower cost than buying options outright: Requires less capital than buying a single call or put option."
        ],
        "Cons": [
            "Limited profit potential: The maximum profit is capped and is usually smaller than the potential loss.",
            "Complex to manage: Requires careful monitoring of the position and potential adjustments as the expiration dates approach.",
            "Volatility Risk: Changes in implied volatility can significantly impact the profitability of the trade."
        ],
        "When to Use": "Expect the underlying asset's price to remain relatively stable or move slightly in a specific direction; want to profit from time decay while limiting risk.",
        "Timeframe Used With": "Near-term options expiring in 2-4 weeks, longer-term options expiring in 1-3 months.",
        "Suitable For": "Traders with a moderate risk tolerance who are comfortable with range-bound trading and have a good understanding of time decay and volatility."
    }
]

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
