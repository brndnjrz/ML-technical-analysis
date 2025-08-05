from src import data_loader
from src import indicators  
from src import plotter
from src import ai_analysis
from src import pdf_generator
from src import config
import streamlit as st
import pandas as pd
import tempfile
import base64
import os

# Set Up Streamlit App UI 
st.set_page_config(page_title="AI Technical Analysis", layout="wide")
st.title("Technical Stock Analysis Dashboard")
st.sidebar.header("⚙️ Configuration")

# --- Enhanced Sidebar Input ---
col1, col2 = st.sidebar.columns(2)
with col1:
    ticker = st.text_input("Stock Ticker:", config.DEFAULT_TICKER)
with col2:
    fetch_realtime = st.checkbox("Real-time Data", True)

start_date = st.sidebar.date_input("Start Date", value=config.DEFAULT_START_DATE)
end_date = st.sidebar.date_input("End Date", value=config.DEFAULT_END_DATE)

# Enhanced interval selection with recommendations
interval_options = {
    "1d": "Daily (Recommended for Long-term)",
    "1h": "Hourly (Good for Short-term)", 
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

# Strategy Selection with Enhanced Options
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
    
    # Specific options strategies
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

# --- STRATEGY-SPECIFIC INDICATOR SELECTION ---
st.sidebar.markdown("### 📈 Technical Indicators")

def get_recommended_indicators(strategy_type, interval):
    """Return recommended indicators based on strategy and timeframe"""
    
    if strategy_type and "Short-Term" in strategy_type:
        return {
            "momentum": ["RSI (14)", "Stochastic (Fast)", "Williams %R"],
            "trend": ["EMA (9)", "EMA (21)", "VWAP"],
            "volatility": ["ATR", "Bollinger Bands (20)", "Implied Volatility"],
            "volume": ["Volume", "OBV", "Money Flow Index"],
            "oscillators": ["MACD (12,26,9)", "CCI"]
        }
    else:  # Long-term
        return {
            "momentum": ["RSI (21)", "RSI (30)", "Stochastic (Slow)"],
            "trend": ["SMA (50)", "EMA (50)", "SMA (200)", "ADX"],
            "volatility": ["ATR (14)", "Bollinger Bands (20)", "Keltner Channels"],
            "volume": ["OBV", "Volume SMA", "Accumulation/Distribution"],
            "oscillators": ["MACD (12,26,9)", "MACD Histogram"]
        }

# Dynamic indicator selection based on strategy
if strategy_type:
    recommended = get_recommended_indicators(strategy_type, interval)
    
    st.sidebar.markdown(f"**Recommended for {strategy_type}:**")
    
    # Organize indicators by category
    selected_indicators = {}
    
    for category, indicators_list in recommended.items():
        st.sidebar.markdown(f"**{category.title()}:**")
        for indicator in indicators_list:
            key = f"{category}_{indicator}"
            # Pre-select most important indicators
            default_value = indicator in ["RSI (14)", "MACD (12,26,9)", "ATR", "Volume"] if "Short-Term" in str(strategy_type) else indicator in ["RSI (21)", "SMA (50)", "ADX", "OBV"]
            selected_indicators[indicator] = st.sidebar.checkbox(indicator, value=default_value, key=key)
    
    # Filter selected indicators
    active_indicators = [k for k, v in selected_indicators.items() if v]
    
else:
    # Fallback to original selection
    active_indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["RSI", "MACD", "ADX", "Stochastic", "OBV", "ATR", "SMA (20)", "EMA (50)", "Bollinger Bands", "VWAP"],
        default=["RSI", "MACD"]
    )

@st.cache_data(ttl=300)
def fetch_and_process_data(ticker, start_date, end_date, interval, strategy_type):
    """Fetch and process data with strategy-specific calculations"""
    try:
        # Fetch data
        data = data_loader.fetch_stock_data(ticker, start_date, end_date, interval)
        
        # Debug: Check what we got back
        print(f"Fetched data type: {type(data)}")
        
        # Handle None return
        if data is None:
            st.error(f"❌ No data available for {ticker}. Please check:")
            st.error("- Ticker symbol is correct")
            st.error("- Date range is valid") 
            st.error("- Market was open during selected period")
            return None, None, None
            
        # Handle empty DataFrame
        if data.empty:
            st.error(f"❌ Empty dataset returned for {ticker}")
            return None, None, None
        
        # Verify it's a DataFrame
        if not isinstance(data, pd.DataFrame):
            st.error(f"❌ Expected DataFrame, got {type(data)}")
            return None, None, None
        
        print(f"✅ Data validation passed: {len(data)} rows, columns: {data.columns.tolist()}")
        
        # Now calculate indicators - make sure this function exists and works
        try:
            data_with_indicators = indicators.calculate_indicators(
                data, 
                timeframe=interval,
                strategy_type=strategy_type,
                selected_indicators=active_indicators
            )
        except Exception as indicator_error:
            st.error(f"❌ Error calculating indicators: {str(indicator_error)}")
            # Return original data if indicator calculation fails
            data_with_indicators = data
        
        # Calculate support/resistance levels
        try:
            levels = indicators.detect_support_resistance(
                data_with_indicators, 
                method="advanced" if "Long-Term" in str(strategy_type) else "quick"
            )
        except Exception as level_error:
            st.error(f"❌ Error detecting levels: {str(level_error)}")
            levels = {'support': [], 'resistance': []}
        
        # Options-specific calculations
        if analysis_type == "Options Trading Strategy":
            try:
                iv_data = indicators.calculate_iv_metrics(ticker, data_with_indicators)
                options_flow = indicators.get_options_flow(ticker) if fetch_realtime else None
                return data_with_indicators, levels, {"iv_data": iv_data, "options_flow": options_flow}
            except Exception as options_error:
                st.warning(f"⚠️ Options data unavailable: {str(options_error)}")
                return data_with_indicators, levels, None
        
        return data_with_indicators, levels, None
        
    except Exception as e:
        st.error(f"❌ Error in fetch_and_process_data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())  # Show full error in streamlit
        return None, None, None


# Fetch Data Button with Enhanced Logic
if st.sidebar.button("🔄 Fetch & Analyze Data", type="primary"):
    with st.spinner("Fetching market data..."):
        data, levels, options_data = fetch_and_process_data(
            ticker, start_date, end_date, interval, strategy_type
        )
        
        if data is not None:
            st.session_state["stock_data"] = data
            st.session_state["levels"] = levels
            st.session_state["options_data"] = options_data
            st.session_state["active_indicators"] = active_indicators
            st.success(f"✅ Data loaded successfully! ({len(data)} candles)")
        else:
            st.error("❌ Failed to fetch data. Please check ticker symbol and date range.")

# --- MAIN ANALYSIS SECTION ---
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]
    levels = st.session_state["levels"]
    options_data = st.session_state.get("options_data")
    
    # Enhanced Support/Resistance Display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price", 
            f"${data['Close'].iloc[-1]:.2f}",
            f"{((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100:.2f}%"
        )
    
    with col2:
        if levels['support']:
            nearest_support = max([s for s in levels['support'] if s < data['Close'].iloc[-1]], default=0)
            st.metric("Nearest Support", f"${nearest_support:.2f}")
    
    with col3:
        if levels['resistance']:
            nearest_resistance = min([r for r in levels['resistance'] if r > data['Close'].iloc[-1]], default=0)
            st.metric("Nearest Resistance", f"${nearest_resistance:.2f}")
    
    # Options-specific metrics
    if options_data and options_data.get("iv_data"):
        st.markdown("### 📊 Stock Metrics")
        iv_cols = st.columns(4)
        
        with iv_cols[0]:
            st.metric("IV Rank", f"{options_data['iv_data'].get('iv_rank', 0):.1f}%")
        with iv_cols[1]:
            st.metric("IV Percentile", f"{options_data['iv_data'].get('iv_percentile', 0):.1f}%")
        with iv_cols[2]:
            st.metric("30-Day HV", f"{options_data['iv_data'].get('hv_30', 0):.1f}%")
        with iv_cols[3]:
            st.metric("VIX Level", f"{options_data['iv_data'].get('vix', 0):.1f}")
    
    # Enhanced Chart with Strategy-Specific Indicators
    st.markdown("### 📈 Technical Analysis Chart")
    
    fig = plotter.create_enhanced_chart(
        data=data,
        indicators=st.session_state["active_indicators"],
        levels=levels,
        strategy_type=strategy_type,
        options_data=options_data,
        interval=interval
    )
    
    st.plotly_chart(fig, use_container_width=True, height=800)
    
    # --- ENHANCED AI ANALYSIS ---
    st.markdown("### 🤖 AI-Powered Strategy Analysis")
    
    analysis_cols = st.columns([2, 1])
    
    with analysis_cols[0]:
        run_analysis = st.button("Run Analysis 💸", type="primary", use_container_width=True)
    
    with analysis_cols[1]:
        confidence_threshold = st.slider("Min Confidence %", 60, 95, 75)
    

    # Build enhanced prompt with strategy-specific focus
    market_context = f"""
    MARKET DATA CONTEXT:
    - Ticker: {ticker}
    - Timeframe: {interval} candles
    - Current Price: ${data['Close'].iloc[-1]:.2f}
    - Strategy Type: {strategy_type}
    - Selected Strategy: {options_strategy}
    - Active Indicators: {', '.join(st.session_state["active_indicators"])}
    
    SUPPORT/RESISTANCE:
    - Support Levels: {['${:.2f}'.format(l) for l in levels['support']]}
    - Resistance Levels: {['${:.2f}'.format(l) for l in levels['resistance']]}
    """
    if options_data:
        market_context += f"""
        
        OPTIONS MARKET DATA:
        - IV Rank: {options_data.get('iv_data', {}).get('iv_rank', 'N/A')}%
        - IV Percentile: {options_data.get('iv_data', {}).get('iv_percentile', 'N/A')}%
        - 30-Day Historical Volatility: {options_data.get('iv_data', {}).get('hv_30', 'N/A')}%
        """
    # Strategy-specific prompts
    if analysis_type == "Options Trading Strategy":
        if "Short-Term" in strategy_type:
            prompt = f"""
            You are an expert short-term options trader. Analyze this {interval} chart for {ticker}.
            
            {market_context}
            
            FOCUS ON SHORT-TERM INDICATORS (1-7 days):
            - Fast RSI signals (overbought >70, oversold <30)
            - MACD crossovers and histogram changes
            - Stochastic momentum shifts
            - ATR for volatility-based strike selection
            - VWAP as intraday support/resistance
            - Volume spikes and unusual activity
            
            PROVIDE:
            1. **TRADE RECOMMENDATION**: [YES/NO] with confidence level
            2. **STRATEGY**: Specific options strategy ({options_strategy})
            3. **STRIKES & EXPIRATION**: Based on ATR and support/resistance
            4. **ENTRY/EXIT CRITERIA**: Specific indicator levels
            5. **RISK MANAGEMENT**: Stop loss and profit targets
            6. **RATIONALE**: Why this setup works for short-term options
            
            Only recommend trades with {confidence_threshold}%+ confidence.
            """
        else:  # Long-term
            prompt = f"""
            You are an expert swing trader specializing in long-term options strategies. 
            
            {market_context}
            
            FOCUS ON TREND INDICATORS (1-3 weeks):
            - RSI divergences and trend strength
            - ADX for trend confirmation (>25 = strong trend)
            - Moving average alignment (50/200 EMA)
            - MACD trend changes and histogram
            - Volume trends and accumulation/distribution
            - Major support/resistance levels
            
            PROVIDE:
            1. **TRADE RECOMMENDATION**: [YES/NO] with confidence level
            2. **STRATEGY**: Specific options strategy ({options_strategy})
            3. **STRIKES & EXPIRATION**: 2-3 weeks out, based on major levels
            4. **TREND ANALYSIS**: Primary trend direction and strength
            5. **RISK/REWARD**: Expected profit targets and stop losses
            6. **RATIONALE**: Why this setup works for swing trading
            
            Only recommend trades with {confidence_threshold}%+ confidence.
            """

    # Run AI analysis synchronously (for simplicity)
    if run_analysis:
        with st.spinner("AI is analyzing the market..."):
            analysis, chart_path = ai_analysis.run_ai_analysis(fig, prompt)
            st.session_state["ai_analysis_result"] = (analysis, chart_path)

    if st.session_state.get("ai_analysis_result") is None and st.session_state.get("ai_analysis_running"):
        st.info("AI analysis started... Please wait.")
        st.spinner("AI is analyzing the market...")

    if st.session_state.get("ai_analysis_result"):
        analysis, chart_path = st.session_state["ai_analysis_result"]
        st.markdown("### 📋 Analysis Results")
        if "YES" in analysis.upper():
            st.success("🟢 **TRADE SIGNAL DETECTED**")
        elif "NO" in analysis.upper():
            st.warning("🔴 **NO TRADE RECOMMENDED**")
        else:
            st.info("🔵 **NEUTRAL SIGNAL**")
        st.markdown(analysis)
        # Enhanced PDF Generation
        if st.button("📄 Generate Detailed Report"):
            with st.spinner("Generating comprehensive report..."):
                pdf = pdf_generator.EnhancedPDF()
                pdf.add_page()
                pdf.add_header(ticker, strategy_type, options_strategy)
                pdf.add_chart(chart_path)
                pdf.add_analysis_text(analysis)
                pdf.add_indicator_summary(data, st.session_state["active_indicators"])
                pdf.add_risk_analysis(data, levels, options_data)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                    pdf.output(tmp_pdf.name)
                    pdf_file_path = tmp_pdf.name
                with open(pdf_file_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                    st.markdown("### 📄 Comprehensive Analysis Report")
                    st.markdown(pdf_display, unsafe_allow_html=True)
                with open(pdf_file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Full Report",
                        data=f.read(),
                        file_name=f'{ticker}_{strategy_type.replace(" ", "_")}_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.pdf',
                        mime="application/pdf"
                    )
                os.remove(chart_path)
                os.remove(pdf_file_path)

        st.session_state["ai_analysis_running"] = False


# --- SIDEBAR: QUICK STATS ---
if "stock_data" in st.session_state:
    st.sidebar.markdown("### 📊 Quick Stats")
    data = st.session_state["stock_data"]
    
    def safe_get_indicator_value(data, indicator_base_name, timeframe="1d"):
        """Safely get indicator value trying different column name patterns"""
        possible_names = [
            indicator_base_name,  # e.g., 'RSI'
            f"{indicator_base_name}_{timeframe}",  # e.g., 'RSI_1d'
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
        """Safely get MACD signal"""
        try:
            # Try different possible column names
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
    
    # Calculate key metrics safely
    current_timeframe = interval  # Use the interval from your app
    
    # RSI
    rsi_current = safe_get_indicator_value(data, "RSI", current_timeframe)
    if rsi_current is not None:
        rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
        st.sidebar.metric("RSI", f"{rsi_current:.1f}", rsi_status)
    else:
        st.sidebar.metric("RSI", "N/A")
    
    # MACD Signal
    macd_signal = safe_get_macd_signal(data, current_timeframe)
    st.sidebar.metric("MACD Signal", macd_signal)
    
    # ATR
    atr_current = safe_get_indicator_value(data, "ATR", current_timeframe)
    if atr_current is not None:
        st.sidebar.metric("ATR (Volatility)", f"${atr_current:.2f}")
    else:
        st.sidebar.metric("ATR (Volatility)", "N/A")


# Footer
st.markdown("---")
st.markdown("*This analysis is for educational purposes only. Always conduct your own research before trading.*")