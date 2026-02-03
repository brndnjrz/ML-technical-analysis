"""Text formatting utilities for analysis reports and display."""
import re
import json
import pandas as pd


def format_analysis_text(text):
    """Clean and format analysis text for better readability in professional report style"""
    if not text:
        return "No analysis available"
    
    # Fix character encoding issues
    text = text.replace('�', '📊')  # Replace broken characters with appropriate emoji
    
    # Clean up markdown formatting
    text = re.sub(r'\*{3,}', '**', text)  # Replace triple asterisks with double
    text = re.sub(r'#{3,}', '### ', text)  # Clean up header formatting
    
    # Fix common formatting issues
    text = text.replace('**:**', ':**')  # Fix double colons
    text = text.replace('- -', '-')  # Fix double dashes
    
    # Format JSON trade parameters into readable format
    def format_trade_params(match):
        json_str = match.group(0)
        try:
            # Clean up the JSON string
            json_str = re.sub(r'^[^{]*{', '{', json_str)
            json_str = re.sub(r'}[^}]*$', '}', json_str)
            params = json.loads(json_str)
            formatted_lines = []
            for key, value in params.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, bool):
                    formatted_value = "✅ Yes" if value else "❌ No"
                elif isinstance(value, (int, float)):
                    if 'price' in key.lower() or 'stop' in key.lower() or 'target' in key.lower():
                        formatted_value = f"${value:.2f}"
                    elif 'period' in key.lower() or 'ma' in key.lower():
                        formatted_value = f"{value} periods"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    if value is not None:
                        formatted_value = str(value).replace('_', ' ').title()
                    else:
                        formatted_value = "Not specified"
                formatted_lines.append(f"* **{formatted_key}:** {formatted_value}")
            return '\n'.join(formatted_lines)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return original text
            return json_str
    
    # Replace JSON blocks with formatted parameters
    text = re.sub(r'\{[^}]*"[^"]*"[^}]*\}', format_trade_params, text)
    
    # Ensure proper spacing around sections
    text = re.sub(r'([🤖📊💡📈👁️⚠️].+?)(\n)([^-•\s])', r'\1\n\n\3', text)
    
    return text.strip()


def format_professional_report(analysis, recommendation, ticker, strategy_type, options_strategy, data, levels, options_data):
    """Format analysis into a professional trade signal report"""
    # Get current market data
    current_price = data['Close'].iloc[-1] if not data.empty else 0
    current_rsi = data.get('RSI', pd.Series([50])).iloc[-1] if 'RSI' in data.columns else 50
    current_macd = data.get('MACD', pd.Series([0])).iloc[-1] if 'MACD' in data.columns else 0
    current_adx = data.get('ADX', pd.Series([25])).iloc[-1] if 'ADX' in data.columns else 25
    current_atr = data.get('ATR', pd.Series([1])).iloc[-1] if 'ATR' in data.columns else 1
    current_vwap = data.get('VWAP', pd.Series([current_price])).iloc[-1] if 'VWAP' in data.columns else current_price
    
    # Get Bollinger Bands
    bb_upper = data.get('BB_upper', pd.Series([current_price * 1.02])).iloc[-1] if 'BB_upper' in data.columns else current_price * 1.02
    bb_lower = data.get('BB_lower', pd.Series([current_price * 0.98])).iloc[-1] if 'BB_lower' in data.columns else current_price * 0.98
    
    # Get support/resistance levels
    nearest_support = max([s for s in levels.get('support', []) if s < current_price], default=current_price * 0.95)
    nearest_resistance = min([r for r in levels.get('resistance', []) if r > current_price], default=current_price * 1.05)
    
    # Get options data
    iv_rank = options_data.get('iv_data', {}).get('iv_rank', 0) if options_data else 0
    iv_percentile = options_data.get('iv_data', {}).get('iv_percentile', 0) if options_data else 0
    vix = options_data.get('iv_data', {}).get('vix', 20) if options_data else 20

    # Extract recommendation details
    action = recommendation.get('action', 'HOLD').upper() if recommendation else 'HOLD'
    confidence = recommendation.get('strategy', {}).get('confidence', 0.5) * 100 if recommendation else 50
    strategy_name = recommendation.get('strategy', {}).get('name', 'Unknown') if recommendation else options_strategy or 'Unknown'

    # Determine risk level
    risk_level = "Low" if iv_rank < 30 and current_atr < current_price * 0.02 else "Medium" if iv_rank < 60 else "High"

    # RSI interpretation
    rsi_status = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
    rsi_signal = "→ Potential bounce up." if current_rsi < 30 else "→ Potential pullback." if current_rsi > 70 else "→ Balanced momentum."

    # MACD interpretation
    macd_status = "Bullish" if current_macd > 0 else "Bearish"
    macd_signal = "(trend continuation confirmed)" if abs(current_macd) > 0.1 else "(weak signal)"

    # ADX interpretation
    trend_strength = "Strong" if current_adx > 25 else "Weak/moderate"

    # Volume analysis
    volume_status = "High" if 'Volume' in data.columns and data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1] else "Normal"
    volume_signal = "(strong participation)" if volume_status == "High" else "(moderate participation)"

    # Calculate stop loss and profit targets
    stop_loss = max(nearest_support, current_price - current_atr * 2)
    profit_target = min(nearest_resistance, current_price + current_atr * 3)
        
    # Format the professional report
    report = f"""# 📊 AI-Powered Stock Analysis Report

**Ticker:** {ticker.upper()}
**Strategy Type:** {strategy_type}
**Options Strategy:** {strategy_name}
**Confidence:** {confidence:.0f}%
**Risk Level:** {risk_level}

---

## 🔎 Market Overview

* **RSI:** {current_rsi:.2f} → {rsi_status} {rsi_signal}
* **MACD:** {macd_status} {macd_signal}
* **Volume:** {volume_status} {volume_signal}
* **ADX (Trend Strength):** {current_adx:.2f} → {trend_strength} trend forming.

---

## 📈 Technical Levels

* **Current Price:** ${current_price:.2f}
* **Support:** ${nearest_support:.2f}
* **Resistance:** ${nearest_resistance:.2f}
* **VWAP:** ${current_vwap:.2f}
* **Bollinger Bands:** ${bb_lower:.2f} – ${bb_upper:.2f}

---

## 🎯 Trade Parameters
"""
    if action in ["BUY", "SELL"]:
        report += f"""
* **Entry Condition:** Price above key technical levels
* **Exit Condition:** Price closes below support or hits target
* **Stop Loss:** ${stop_loss:.2f} (${stop_loss - current_price:.2f} from current)
* **Profit Target:** ${profit_target:.2f} (+${profit_target - current_price:.2f} upside)
* **Trailing Stop:** Active (lock in gains if price rises)
"""
    else:
        report += "\n* _No actionable trade parameters for HOLD recommendation._\n"

    # Prepare text for risk assessment
    upside_text = "limited upside" if current_rsi > 70 else "potential upside" if current_rsi < 30 else "balanced risk/reward"
    atr_size_text = "small" if current_atr < current_price * 0.015 else "moderate" if current_atr < current_price * 0.03 else "large"
    vix_status_text = "low" if vix < 15 else "moderate" if vix < 25 else "high"
    
    report += f"""
---

## ⚖️ Risk Assessment

* **RSI Level:** {rsi_status} suggests {upside_text}.
* **Volatility Risk:** {risk_level} → IV Rank {iv_rank:.1f}%, IV Percentile {iv_percentile:.1f}%.
* **ATR (Daily Move):** ${current_atr:.2f} → expect {atr_size_text} daily swings.
* **VIX:** {vix:.1f} → market-wide volatility {vix_status_text}.

---

## ✅ Recommendation

"""
    
    # Determine recommendation text
    action_explanation = "Trend-following setup supports" if action == "BUY" else "Technical signals suggest" if action == "SELL" else "Neutral signals recommend"
    
    report += f"* **{action}:** {action_explanation} a {action.lower()} position."

    if action in ["BUY", "SELL"]:
        report += f"\n* **Stop:** Place {'below' if action == 'BUY' else 'above'} ${stop_loss:.2f} (to limit downside)."
        report += f"\n* **Take Profit:** ${profit_target:.2f} zone."
        options_play = "Call strategies preferred" if action == "BUY" else "Put strategies preferred"
        iv_env = "low" if iv_rank < 30 else "high"
        report += f"\n* **Options Play:** {options_play} in {iv_env} IV environment."
    else:
        report += "\n* _No stop loss, take profit, or options play for HOLD recommendation._"

    # Add Risk Warning section
    report += "\n\n---\n\n"
    report += "## ⚠️ Risk Warning\n\n"
    report += "This is AI-generated analysis for **educational purposes only**.\n"
    report += "Always perform your own due diligence. Not financial advice.\n\n"
    report += "---"
    return report