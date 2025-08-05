import streamlit as st
import pandas as pd
from src import plotter, ai_analysis, config
from src.data_pipeline import fetch_and_process_data
from src.prediction import get_fundamental_metrics, predict_next_day_close
from src.pdf_utils import generate_and_display_pdf
from src.ui_components import render_sidebar_quick_stats, sidebar_config, sidebar_indicator_selection


# Set Up Streamlit App UI 
st.set_page_config(page_title="AI Technical Analysis", layout="wide")
st.title("Technical Stock Analysis Dashboard")
st.sidebar.header("⚙️ Configuration")

# --- Modular Sidebar ---
# Stock ticker, date range, timeframee/interval, analysis type, strategy type, technical indicators
ticker, fetch_realtime, start_date, end_date, interval, analysis_type, strategy_type, options_strategy = sidebar_config(config)
active_indicators = sidebar_indicator_selection(strategy_type, interval)



# Fetch Data Button with Enhanced Logic
# This button fetches data based on user inputs and updates the session state
if st.sidebar.button("🔄 Fetch & Analyze Data", type="primary"):
    with st.spinner("Fetching market data..."):
        data, levels, options_data = fetch_and_process_data(
            ticker, start_date, end_date, interval, strategy_type, analysis_type, fetch_realtime, active_indicators
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
# This section displays the stock data, technical indicators, and analysis results
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




    fundamentals = get_fundamental_metrics(ticker)

    st.markdown("### 📊 Stock Metrics")
    metric_labels = [
        ("IV Rank", f"{options_data['iv_data'].get('iv_rank', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("IV Percentile", f"{options_data['iv_data'].get('iv_percentile', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("30-Day HV", f"{options_data['iv_data'].get('hv_30', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("VIX Level", f"{options_data['iv_data'].get('vix', 0):.1f}" if options_data and options_data.get("iv_data") else "N/A"),
        ("EPS", fundamentals.get("EPS", "N/A")),
        ("P/E Ratio", fundamentals.get("P/E Ratio", "N/A")),
        ("Revenue Growth", fundamentals.get("Revenue Growth", "N/A")),
        ("Profit Margin", fundamentals.get("Profit Margin", "N/A")),
    ]

    cols = st.columns(4)
    for i, (label, value) in enumerate(metric_labels):
        with cols[i % 4]:
            st.metric(label, value if value is not None else "N/A")


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
            # NEW: Get the price prediction
            predicted_price = predict_next_day_close(data.copy(), fundamentals) # Pass a copy to avoid modifying the original
            if predicted_price is not None:
                # NEW: Add predicted price to the data
                data['Predicted_Close'] = predicted_price
                price_change = predicted_price - data['Close'].iloc[-1]
                data['Predicted_Price_Change'] = price_change
                # Update market context with predicted price
                market_context += f"\nNEXT DAY PREDICTED CLOSE: ${predicted_price:.2f} (Change: ${price_change:.2f})\n"

                # NEW: Update the prompt to consider the prediction
                if "Short-Term" in strategy_type:
                    prompt = prompt.replace("PROVIDE:", f"""PREDICTED NEXT DAY CLOSE: ${predicted_price:.2f}.
CONSIDER HOW THIS PRICE AFFECTS SHORT-TERM INDICATORS AND MOMENTUM.

PROVIDE:""")
                else:
                    prompt = prompt.replace("PROVIDE:", f"""PREDICTED NEXT DAY CLOSE: ${predicted_price:.2f}.
CONSIDER HOW THIS PRICE AFFECTS TRENDS AND LONG-TERM STRATEGIES.

PROVIDE:""")

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
                generate_and_display_pdf(
                    ticker, strategy_type, options_strategy, data, analysis, chart_path, levels, options_data, st.session_state["active_indicators"]
                )

        st.session_state["ai_analysis_running"] = False



# --- SIDEBAR: QUICK STATS ---
if "stock_data" in st.session_state:
    render_sidebar_quick_stats(st.session_state["stock_data"], interval)


# Footer
st.markdown("---")
st.markdown("*This analysis is for educational purposes only. Always conduct your own research before trading.*")