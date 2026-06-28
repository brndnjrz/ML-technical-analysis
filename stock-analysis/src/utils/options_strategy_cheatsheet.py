"""
Options Strategy Cheatsheet for AI Model Reference
Professional analyst framework for options strategy selection and analysis
"""

from typing import Dict, Any, List
import pandas as pd

# Professional Options Strategy Cheatsheet for AI models
# This structured framework helps the AI make more informed options strategy recommendations
OPTIONS_STRATEGY_CHEATSHEET = {
    "step1_market_trend": {
        "description": "Identify Market Trend (Bias)",
        "indicators": {
            "moving_averages": [
                "Price above 50 & 200 → Uptrend confirmed",
                "Price below 50 & 200 → Downtrend confirmed",
                "20 crossing above 50 → Bullish crossover",
                "20 crossing below 50 → Bearish crossover"
            ],
            "macd": [
                "MACD > Signal → Bullish momentum",
                "MACD < Signal → Bearish momentum",
                "Histogram expanding → Strong trend continuation"
            ],
            "rsi": [
                "50-70 = Uptrend strength",
                "30-50 = Downtrend pressure",
                "> 70 = Overbought, < 30 = Oversold (watch for reversals)"
            ]
        },
        "conclusion": "Trend is the foundation — don't fight the market. Calls in uptrend, Puts in downtrend, Neutral plays in sideways chop."
    },
    
    "step2_volatility_check": {
        "description": "Volatility Check (IV & Risk)",
        "indicators": {
            "iv_rank_percentile": [
                "IV Rank < 20 → Options cheap (buy premium)",
                "IV Rank > 50 → Options expensive (sell premium)"
            ],
            "vix": [
                "< 15 → Calm market (low IV)",
                "> 20 → Rising fear (high IV)"
            ],
            "bollinger_band_width": [
                "Tight bands → Low volatility, breakout incoming",
                "Wide bands → High volatility, often mean-reverting"
            ]
        },
        "conclusion": "Option prices are volatility-driven. Low IV = buy options (cheap). High IV = sell options (expensive)."
    },
    
    "step3_strength_momentum": {
        "description": "Confirm Strength & Momentum",
        "indicators": {
            "adx": [
                "> 20 = Trend forming",
                "> 25 = Strong trend",
                "< 20 = Weak/choppy"
            ],
            "volume_obv": [
                "Rising volume on breakout = confirmation",
                "Falling volume = weak move"
            ],
            "atr": [
                "Higher ATR = bigger daily swings → more profit potential for options",
                "Low ATR = stock too slow → not worth option premium"
            ]
        },
        "conclusion": "Options need movement. ADX + volume confirm strength. ATR confirms payoff potential."
    },
    
    "step4_support_resistance": {
        "description": "Support & Resistance",
        "tools": [
            "Horizontal zones (price history)",
            "Fibonacci retracements (38.2%, 50%, 61.8%)",
            "Pivot Points",
            "Trendlines / Channels"
        ],
        "conclusion": "Calls near support = great risk/reward. Puts near resistance = great risk/reward. Breakouts/breakdowns = strong continuation setups."
    },
    
    "step5_strategy_selection": {
        "description": "Strategy Selection Matrix",
        "matrix": {
            "bullish_low_iv": {
                "strategy": "Long Calls / Bull Call Debit Spread",
                "rationale": "Cheap premium, ride uptrend",
                "Description": "Buy calls or a bull call spread when expecting a strong upward move and implied volatility is low. Profit if the stock rises above the strike(s)."
            },
            "bullish_high_iv": {
                "strategy": "Bull Put Credit Spread / Covered Calls",
                "rationale": "Collect inflated premiums",
                "Description": "Sell bull put spreads or covered calls when bullish but IV is high. Profit from premium decay if the stock stays above the short strike."
            },
            "bearish_low_iv": {
                "strategy": "Long Puts / Bear Put Debit Spread",
                "rationale": "Cheap downside bet",
                "Description": "Buy puts or a bear put spread when expecting a significant downward move and IV is low. Profit if the stock falls below the strike(s)."
            },
            "bearish_high_iv": {
                "strategy": "Bear Call Credit Spread",
                "rationale": "Collect premium while stock drifts lower",
                "Description": "Sell bear call spreads when bearish and IV is high. Profit if the stock stays below the short call strike."
            },
            "neutral_low_iv": {
                "strategy": "Straddle / Strangle",
                "rationale": "Bet on volatility expansion",
                "Description": "Buy straddles or strangles when expecting a big move in either direction and IV is low. Profit if the stock moves far from the strikes."
            },
            "neutral_high_iv": {
                "strategy": "Iron Condor",
                "rationale": "Profit if stock stays in range",
                "Description": "Sell iron condors when expecting little movement and IV is high. Profit if the stock stays between the short strikes and options decay."
            }
        }
    },
    
    "step6_risk_management": {
        "description": "Risk Management Rules (Non-Negotiable)",
        "rules": [
            "Stop loss: Just below support (for calls) / above resistance (for puts)",
            "Position sizing: Max 1-2% of portfolio risk per trade",
            "Reward-to-risk ratio: Minimum 2:1",
            "Trailing stop: Use if trend is strong to lock in profits"
        ],
        "conclusion": "Winning strategies are useless if risk isn't controlled. Pros survive on risk control, not just predictions."
    },
    
    "full_workflow": [
        "Check Trend → MA + MACD + RSI",
        "Check Volatility → IV Rank, VIX, Bollinger Bands",
        "Check Momentum → ADX, Volume, ATR",
        "Check Levels → Support, Resistance, Fibonacci",
        "Choose Strategy → Based on Trend + IV",
        "Manage Risk → Stops, sizing, R:R"
    ]
}

def get_option_strategy_recommendation(data: pd.DataFrame, options_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an options strategy recommendation based on market data and the pro analyst cheatsheet
    
    Args:
        data: DataFrame containing price and indicator data
        options_data: Dictionary containing options-related data (IV rank, etc.)
        
    Returns:
        Dictionary containing recommended strategy and rationale
    """
    recommendation = {
        "strategy": "",
        "rationale": "",
        "risk_assessment": "",
        "key_levels": {}
    }
    
    # This function would be fully implemented to analyze the data against the cheatsheet rules
    # and return a recommendation based on the current market conditions
    
    return recommendation

def get_analyst_cheatsheet_markdown() -> str:
    """
    Returns the options strategy cheatsheet as a formatted markdown string for display
    """
    cheatsheet = """
# 📊 Options Strategy Analyst Cheatsheet (Full Depth)

## 1️⃣ Step One: Identify Market Trend (Bias)

### 🔑 Indicators to Check:

* **Moving Averages (20, 50, 200):**
  * Price above 50 & 200 → Uptrend confirmed.
  * Price below 50 & 200 → Downtrend confirmed.
  * 20 crossing above 50 → Bullish crossover.
  * 20 crossing below 50 → Bearish crossover.
* **MACD (12,26,9):**
  * MACD > Signal → Bullish momentum.
  * MACD < Signal → Bearish momentum.
  * Histogram expanding → Strong trend continuation.
* **RSI (14):**
  * 50–70 = Uptrend strength.
  * 30–50 = Downtrend pressure.
  * > 70 = Overbought, < 30 = Oversold (watch for reversals).

📌 **Why it matters:** Trend is the foundation — you don't want to fight the market.
* Calls = in uptrend.
* Puts = in downtrend.
* Neutral plays = in sideways chop.

## 2️⃣ Step Two: Volatility Check (IV & Risk)

### 🔑 Indicators to Check:

* **IV Rank / IV Percentile:**
  * IV Rank < 20 → Options cheap (buy premium).
  * IV Rank > 50 → Options expensive (sell premium).
* **VIX (for SPY proxy):**
  * < 15 → Calm market (low IV).
  * > 20 → Rising fear (high IV).
* **Bollinger Band Width:**
  * Tight bands → Low volatility, breakout incoming.
  * Wide bands → High volatility, often mean-reverting.

📌 **Why it matters:** Option prices are volatility-driven.
* Low IV = buy options (cheap).
* High IV = sell options (expensive).

## 3️⃣ Step Three: Confirm Strength & Momentum

### 🔑 Indicators to Check:

* **ADX (14):**
  * > 20 = Trend forming.
  * > 25 = Strong trend.
  * < 20 = Weak/choppy.
* **Volume & OBV (On Balance Volume):**
  * Rising volume on breakout = confirmation.
  * Falling volume = weak move.
* **ATR (14):**
  * Higher ATR = bigger daily swings → more profit potential for options.
  * Low ATR = stock too slow → not worth option premium.

📌 **Why it matters:** Options need movement. ADX + volume confirm strength. ATR confirms payoff potential.

## 4️⃣ Step Four: Support & Resistance

### 🔑 Tools to Check:

* Horizontal zones (price history).
* Fibonacci retracements (38.2%, 50%, 61.8%).
* Pivot Points.
* Trendlines / Channels.

📌 **Why it matters:**
* Calls near support = great risk/reward.
* Puts near resistance = great risk/reward.
* Breakouts past resistance or breakdowns past support = strong continuation setups.

## 5️⃣ Strategy Selection Matrix

| Market Trend | IV Level | Best Strategy                          | Why It Works                             |
| ------------ | -------- | -------------------------------------- | ---------------------------------------- |
| **Bullish**  | Low IV   | Long Calls / Bull Call Debit Spread    | Cheap premium, ride uptrend              |
| **Bullish**  | High IV  | Bull Put Credit Spread / Covered Calls | Collect inflated premiums                |
| **Bearish**  | Low IV   | Long Puts / Bear Put Debit Spread      | Cheap downside bet                       |
| **Bearish**  | High IV  | Bear Call Credit Spread                | Collect premium while stock drifts lower |
| **Neutral**  | Low IV   | Straddle / Strangle                    | Bet on volatility expansion              |
| **Neutral**  | High IV  | Iron Condor                            | Profit if stock stays in range           |

## 6️⃣ Risk Management Rules (Non-Negotiable)

* **Stop loss:** Just below support (for calls) / above resistance (for puts).
* **Position sizing:** Max 1–2% of portfolio risk per trade.
* **Reward-to-risk ratio:** Minimum 2:1.
* **Trailing stop:** Use if trend is strong to lock in profits.

📌 **Why it matters:** Winning strategies are useless if risk isn't controlled. Pros survive on risk control, not just predictions.

# 🔁 Full Workflow (Analyst Playbook)

1. **Check Trend** → MA + MACD + RSI.
2. **Check Volatility** → IV Rank, VIX, Bollinger Bands.
3. **Check Momentum** → ADX, Volume, ATR.
4. **Check Levels** → Support, Resistance, Fibonacci.
5. **Choose Strategy** → Based on Trend + IV.
6. **Manage Risk** → Stops, sizing, R:R.
"""
    return cheatsheet
