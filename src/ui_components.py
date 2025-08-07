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
        ["Options Trading Strategy", "Stock Buy/Hold/Sell", "Multi-Strategy Analysis"]
    )
    strategy_type = None
    options_strategy = None
    if analysis_type == "Options Trading Strategy":
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

def sidebar_indicator_selection(strategy_type, interval):
    st.sidebar.markdown("### 📈 Technical Indicators")
    def get_recommended_indicators(strategy_type, interval):
        if strategy_type and "Short-Term" in strategy_type:
            return {
                "momentum": ["RSI (14)", "Stochastic (Fast)", "Williams %R"],
                "trend": ["EMA (9)", "EMA (21)", "VWAP"],
                "volatility": ["ATR", "Bollinger Bands (20)", "Implied Volatility"],
                "volume": ["Volume", "OBV", "Money Flow Index"],
                "oscillators": ["MACD (12,26,9)", "CCI"]
            }
        else:
            return {
                "momentum": ["RSI (21)", "RSI (30)", "Stochastic (Slow)"],
                "trend": ["SMA (50)", "EMA (50)", "SMA (200)", "ADX"],
                "volatility": ["ATR (14)", "Bollinger Bands (20)", "Keltner Channels"],
                "volume": ["OBV", "Volume SMA", "Accumulation/Distribution"],
                "oscillators": ["MACD (12,26,9)", "MACD Histogram"]
            }
    if strategy_type:
        recommended = get_recommended_indicators(strategy_type, interval)
        st.sidebar.markdown(f"**Recommended for {strategy_type}:**")
        selected_indicators = {}
        for category, indicators_list in recommended.items():
            st.sidebar.markdown(f"**{category.title()}:**")
            for indicator in indicators_list:
                key = f"{category}_{indicator}"
                default_value = indicator in ["RSI (14)", "MACD (12,26,9)", "ATR", "Volume"] if "Short-Term" in str(strategy_type) else indicator in ["RSI (21)", "SMA (50)", "ADX", "OBV"]
                selected_indicators[indicator] = st.sidebar.checkbox(indicator, value=default_value, key=key)
        active_indicators = [k for k, v in selected_indicators.items() if v]
    else:
        active_indicators = st.sidebar.multiselect(
            "Select Indicators:",
            ["RSI", "MACD", "ADX", "Stochastic", "OBV", "ATR", "SMA (20)", "EMA (50)", "Bollinger Bands", "VWAP"],
            default=["RSI", "MACD"]
        )
    return active_indicators


def render_sidebar_quick_stats(data, interval):
    def safe_get_indicator_value(data, indicator_base_name, timeframe="1d"):
        possible_names = [
            indicator_base_name,
            f"{indicator_base_name}_{timeframe}",
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
    st.sidebar.markdown("### 📊 Quick Stats")
    current_timeframe = interval
    rsi_current = safe_get_indicator_value(data, "RSI", current_timeframe)
    if rsi_current is not None:
        rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
        st.sidebar.metric("RSI", f"{rsi_current:.1f}", rsi_status)
    else:
        st.sidebar.metric("RSI", "N/A")
    macd_signal = safe_get_macd_signal(data, current_timeframe)
    st.sidebar.metric("MACD Signal", macd_signal)
    atr_current = safe_get_indicator_value(data, "ATR", current_timeframe)
    if atr_current is not None:
        st.sidebar.metric("ATR (Volatility)", f"${atr_current:.2f}")
    else:
        st.sidebar.metric("ATR (Volatility)", "N/A")
