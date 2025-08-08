import streamlit as st
import pandas as pd



def sidebar_config(config):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker:", config.DEFAULT_TICKER).upper().strip()
    with col2:
        # fetch_realtime = st.checkbox("Real-time Data", True)
        start_date = st.sidebar.date_input("Start Date", value=config.DEFAULT_START_DATE)
        end_date = st.sidebar.date_input("End Date", value=config.DEFAULT_END_DATE)
    interval_options = {
        "1d": "Daily (Long-term)",
        "1h": "Hourly (Short-term)",
        "30m": "30min (Short-term Scalping)",
        "15m": "15min (Day Trading)",
        "5m": "5min (Scalping Only)"
    }
    interval = st.sidebar.selectbox(
        "Select Timeframe:",
        options=list(interval_options.keys()),
        format_func=lambda x: interval_options[x],
        index=0
    )
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Options Trading Strategy", "Stock Buy/Hold/Sell"]
    )
    strategy_type = None
    options_strategy = None
    
    # Set strategy type for all analysis types
    if analysis_type in ["Options Trading Strategy", "Stock Buy/Hold/Sell"]:
        strategy_type = st.sidebar.selectbox(
            "Trading Timeframe:",
            ["Short-Term (1-7 days)", "Long-Term (1-3 weeks)", "Custom"],
            index=1
        )
        if "Short-Term" in strategy_type:
            options_strategy = st.sidebar.selectbox(
                "Short-Term Strategy:",
                ["Day Trading Calls/Puts", "Iron Condor", "Straddle/Strangle",
                 "Butterfly Spread", "Scalping Options"]
            )
        elif "Long-Term" in strategy_type:
            options_strategy = st.sidebar.selectbox(
                "Long-Term Strategy:",
                ["Swing Trading", "Covered Calls", "Protective Puts",
                 "Vertical Spreads", "Calendar Spreads"]
            )
    return ticker, start_date, end_date, interval, analysis_type, strategy_type, options_strategy

def sidebar_indicator_selection(strategy_type, interval, data=None):
    st.sidebar.markdown("### 📈 AI-Selected Technical Indicators")
    
    def analyze_market_conditions(data):
        if data is None:
            return "unknown"
        
        try:
            # Calculate basic volatility
            if 'Close' in data.columns:
                returns = data['Close'].pct_change()
                volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                
                # Determine market condition
                if volatility > 0.4:  # High volatility
                    return "high_volatility"
                elif volatility < 0.15:  # Low volatility
                    return "low_volatility"
                else:
                    return "normal_volatility"
        except:
            return "normal_volatility"
    
    def get_ai_recommended_indicators(strategy_type, interval, market_condition):
        base_indicators = {
            "momentum": [],
            "trend": [],
            "volatility": [],
            "volume": [],
            "oscillators": [],
            "prediction": []
        }
        
        def get_timeframe_indicators(is_short_term):
            # Helper function to get appropriate indicators based on timeframe
            if is_short_term:
                return {
                    # Short-term indicators (15m timeframe focused)
                    "rsi": ["RSI (15min)", "RSI (14)"],
                    "ma": ["EMA (20) 15m", "EMA (50) 15m", "SMA (20) 15m"],
                    "volatility": ["ATR 15m", "Bollinger Bands 15m"],
                    "momentum": ["MACD 15m", "Stochastic Fast 15m"],
                    "volume": ["OBV 15m", "VWAP 15m"],
                    "trend": ["ADX 15m"]
                }
            else:
                return {
                    # Long-term indicators (regular timeframe)
                    "rsi": ["RSI (14)", "RSI (21)"],
                    "ma": ["SMA (20)", "SMA (50)", "EMA (20)", "EMA (50)"],
                    "volatility": ["ATR", "Bollinger Bands"],
                    "momentum": ["MACD", "MACD Line", "Signal Line"],
                    "volume": ["OBV", "VWAP", "Volume"],
                    "trend": ["ADX"]
                }
        
        # Get appropriate indicators based on strategy type
        is_short_term = strategy_type and "Short-Term" in strategy_type
        indicators = get_timeframe_indicators(is_short_term)
        
        # Base indicator selection
        base_indicators.update({
            "momentum": indicators["rsi"],
            "trend": indicators["ma"] + [indicators["trend"][0]],
            "volatility": indicators["volatility"],
            "volume": indicators["volume"],
            "oscillators": indicators["momentum"],
            "prediction": ["Predicted Close", "Predicted Price Change"]
        })
        
        # Add market condition specific indicators
        if market_condition == "high_volatility":
            # High volatility - focus on momentum and quick moves
            if is_short_term:
                base_indicators["momentum"].extend(["RSI (15min)", "Stochastic Fast 15m"])
                base_indicators["volatility"].extend(["ATR 15m"])
                base_indicators["volume"].extend(["VWAP 15m"])
            else:
                base_indicators["momentum"].extend(["RSI (14)", "MACD"])
                base_indicators["volatility"].extend(["ATR"])
                base_indicators["volume"].extend(["VWAP", "OBV"])
                
        elif market_condition == "low_volatility":
            # Low volatility - focus on trend following
            if is_short_term:
                base_indicators["trend"].extend(["EMA (20) 15m", "ADX 15m"])
                base_indicators["volume"].extend(["OBV 15m"])
            else:
                base_indicators["trend"].extend(["SMA (50)", "EMA (50)"])
                base_indicators["volume"].extend(["OBV", "Volume"])
                
        # Add prediction indicators for all scenarios
        if is_short_term:
            base_indicators["prediction"].extend(["Volatility", "Returns"])
        
        return base_indicators
    
    market_condition = analyze_market_conditions(data)
    recommended = get_ai_recommended_indicators(strategy_type, interval, market_condition)
    
    # Display AI-selected indicators
    st.sidebar.markdown(f"**AI-Selected Indicators for {strategy_type if strategy_type else 'Analysis'}**")
    st.sidebar.markdown(f"*Market Condition: {market_condition.replace('_', ' ').title()}*")
    
    active_indicators = []
    for category, indicators in recommended.items():
        if indicators:  # Only show categories with indicators
            st.sidebar.markdown(f"**{category.title()}:**")
            for indicator in indicators:
                st.sidebar.markdown(f"- {indicator}")
                active_indicators.append(indicator)
    
    # Add explanation for selections
    with st.sidebar.expander("🤖 Why these indicators?"):
        if "Short-Term" in str(strategy_type):
            st.markdown("""
            **Short-Term Trading Indicators:**
            
            🔄 **15-Minute Timeframe Focus**
            - RSI and Stochastic for quick momentum shifts
            - Short-term EMAs for immediate trend
            - VWAP for intraday price levels
            
            📊 **Risk Management**
            - ATR for volatility measurement
            - Bollinger Bands for price channels
            - Volume analysis for confirmation
            
            🎯 **Entry/Exit Signals**
            - MACD for momentum shifts
            - Stochastic for overbought/oversold
            - Volume for confirmation
            
            🔮 **AI Predictions**
            - Price predictions for next moves
            - Volatility forecasting
            - Return projections
            """)
        else:
            st.markdown("""
            **Long-Term Trading Indicators:**
            
            📈 **Trend Analysis**
            - Multiple timeframe EMAs/SMAs
            - ADX for trend strength
            - MACD for trend confirmation
            
            📊 **Risk Assessment**
            - Standard ATR for volatility
            - Regular Bollinger Bands
            - Volume trends
            
            💡 **Strategic Insights**
            - RSI divergences
            - Price action patterns
            - Volume-price relationships
            
            🔮 **AI Predictions**
            - Longer-term price targets
            - Trend continuation probability
            - Market condition analysis
            """)
    
    return active_indicators


def render_sidebar_quick_stats(data, interval):
    def safe_get_indicator_value(data, indicator_base_name, timeframe="1d"):
        possible_names = [
            indicator_base_name,
            f"{indicator_base_name}_{timeframe}",
            indicator_base_name.replace("_", "")  # Try without underscore
        ]
        for name in possible_names:
            if name in data.columns:
                try:
                    value = data[name].iloc[-1]
                    if pd.notnull(value):
                        return value
                except:
                    continue
        return None
    def safe_get_macd_signal(data, timeframe="1d"):
        try:
            possible_macd = ['MACD', f'MACD_{timeframe}']
            possible_signal = ['MACD_Signal', f'MACD_Signal_{timeframe}']
            macd_value = None
            signal_value = None
            for col in possible_macd:
                if col in data.columns:
                    macd_value = data[col].iloc[-1]
                    break
            for col in possible_signal:
                if col in data.columns:
                    signal_value = data[col].iloc[-1]
                    break
            if macd_value is not None and signal_value is not None:
                if pd.notnull(macd_value) and pd.notnull(signal_value):
                    return "Bullish" if macd_value > signal_value else "Bearish"
            return "N/A"
        except Exception as e:
            print(f"Error calculating MACD signal: {e}")
            return "N/A"
    st.sidebar.markdown("### 📊 Technical Indicators")
    current_timeframe = interval

    # RSI Section
    st.sidebar.markdown("**Momentum**")
    rsi_current = safe_get_indicator_value(data, "RSI_14", current_timeframe) or safe_get_indicator_value(data, "RSI", current_timeframe)
    if rsi_current is not None:
        rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
        st.sidebar.metric("RSI (14)", f"{rsi_current:.1f}", rsi_status)
    else:
        st.sidebar.metric("RSI (14)", "N/A")

    # MACD Section
    st.sidebar.markdown("**Trend**")
    macd_signal = safe_get_macd_signal(data, current_timeframe)
    st.sidebar.metric("MACD Signal", macd_signal)

    # Moving Averages
    sma_20 = safe_get_indicator_value(data, "SMA_20", current_timeframe)
    sma_50 = safe_get_indicator_value(data, "SMA_50", current_timeframe)
    if sma_20 is not None and sma_50 is not None:
        ma_trend = "Bullish" if sma_20 > sma_50 else "Bearish"
        st.sidebar.metric("MA Trend", ma_trend, f"SMA20: ${sma_20:.2f}")

    # Volatility Section
    st.sidebar.markdown("**Volatility**")
    atr_current = safe_get_indicator_value(data, "ATR", current_timeframe)
    if atr_current is not None:
        st.sidebar.metric("ATR", f"${atr_current:.2f}")

    bb_middle = safe_get_indicator_value(data, "BB_middle", current_timeframe)
    bb_width = None
    if bb_middle is not None:
        bb_upper = safe_get_indicator_value(data, "BB_upper", current_timeframe)
        bb_lower = safe_get_indicator_value(data, "BB_lower", current_timeframe)
        if bb_upper is not None and bb_lower is not None:
            bb_width = (bb_upper - bb_lower) / bb_middle
            st.sidebar.metric("BB Width", f"{bb_width:.2%}")

    # Volume Section
    st.sidebar.markdown("**Volume**")
    vwap = safe_get_indicator_value(data, "VWAP", current_timeframe)
    if vwap is not None:
        current_price = data['Close'].iloc[-1] if 'Close' in data.columns else None
        if current_price is not None:
            vwap_signal = "Above VWAP" if current_price > vwap else "Below VWAP"
            st.sidebar.metric("VWAP", f"${vwap:.2f}", vwap_signal)
