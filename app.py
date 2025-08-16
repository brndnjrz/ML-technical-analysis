import streamlit as st
import pandas as pd
import os
import logging 
import re

# Core functionality imports
from src import plotter
from src.core.data_pipeline import fetch_and_process_data
from src.analysis.prediction import get_fundamental_metrics, predict_next_day_close
from src.analysis.ai_analysis import run_ai_analysis

# Utility imports
from src.utils.config import DEFAULT_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE
from src.utils.logging_config import setup_logging, set_log_level
from src.utils.temp_manager import temp_manager, cleanup_old_temp_files
from src.pdf_utils import generate_and_display_pdf

# UI components
from src.ui_components import (
    render_sidebar_quick_stats, 
    sidebar_config, 
    sidebar_indicator_selection
)

# Data
from src.trading_strategies import strategies_data, get_strategy_by_name

# Setup cleaner logging for Streamlit
setup_logging(level=logging.INFO, enable_file_logging=False)

# Suppress verbose libraries
logging.getLogger('kaleido').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Clean up old temp files on startup
cleanup_old_temp_files()

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
        import json
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
    try:
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

* **Entry Condition:** Price above key technical levels
* **Exit Condition:** Price closes below support or hits target
* **Stop Loss:** ${stop_loss:.2f} (${stop_loss - current_price:.2f} from current)
* **Profit Target:** ${profit_target:.2f} (+${profit_target - current_price:.2f} upside)
* **Trailing Stop:** Active (lock in gains if price rises)

---

## ⚖️ Risk Assessment

* **RSI Level:** {rsi_status} suggests {"limited upside" if current_rsi > 70 else "potential upside" if current_rsi < 30 else "balanced risk/reward"}.
* **Volatility Risk:** {risk_level} → IV Rank {iv_rank:.1f}%, IV Percentile {iv_percentile:.1f}%.
* **ATR (Daily Move):** ${current_atr:.2f} → expect {"small" if current_atr < current_price * 0.015 else "moderate" if current_atr < current_price * 0.03 else "large"} daily swings.
* **VIX:** {vix:.1f} → market-wide volatility {"low" if vix < 15 else "moderate" if vix < 25 else "high"}.

---

## ✅ Recommendation

* **{action}:** {"Trend-following setup supports" if action == "BUY" else "Technical signals suggest" if action == "SELL" else "Neutral signals recommend"} a {action.lower()} position.
* **Stop:** Place {'below' if action == 'BUY' else 'above'} ${stop_loss:.2f} (to limit downside).
* **Take Profit:** ${profit_target:.2f} zone.
* **Options Play:** {"Call strategies preferred" if action == "BUY" else "Put strategies preferred" if action == "SELL" else "Neutral strategies recommended"} in {"low" if iv_rank < 30 else "high"} IV environment.

---

## ⚠️ Risk Warning

This is AI-generated analysis for **educational purposes only**.
Always perform your own due diligence. Not financial advice.

---
"""
        
        return report
        
    except Exception as e:
        print(f"Error formatting professional report: {e}")
        return format_analysis_text(analysis) if analysis else "Report formatting error occurred."

# Set Up Streamlit App UI 
st.set_page_config(page_title="AI Technical Analysis", layout="wide")
st.title("Technical Stock Analysis Dashboard")
st.sidebar.header("⚙️ Configuration")

# --- Logging Level Selector ---
with st.sidebar.expander("🔧 Debug Settings"):
    log_level = st.selectbox(
        "Log Level",
        ["INFO", "DEBUG", "WARNING", "ERROR"],
        index=0,
        help="Control console message verbosity"
    )
    if st.button("Apply Log Level"):
        set_log_level(log_level)
        st.success(f"Log level set to {log_level}")

# --- Vision Analysis Settings ---
st.sidebar.markdown("### 👁️ Vision Analysis Settings")
vision_timeout = st.sidebar.slider(
    "Vision Analysis Timeout (seconds)",
    min_value=30,
    max_value=5000,
    value=90,
    step=30,
    help="Adjust timeout for AI vision analysis. Increase if you experience frequent timeouts."
)

if vision_timeout > 2000:
    st.sidebar.warning("⚠️ Long timeouts may slow down analysis")
elif vision_timeout < 60:
    st.sidebar.info("ℹ️ Short timeouts may cause analysis failures")

enable_vision_analysis = st.sidebar.checkbox(
    "Enable Vision Analysis", 
    value=True, 
    help="Uncheck to skip chart vision analysis and speed up processing"
)

# --- Modular Sidebar ---
# Stock ticker, date range, timeframee/interval, analysis type, strategy type, technical indicators
ticker, start_date, end_date, interval, analysis_type, strategy_type, options_strategy, options_priority = sidebar_config()

# Store analysis type in session state for other components to access
if 'analysis_type' not in st.session_state:
    st.session_state['analysis_type'] = analysis_type
else:
    st.session_state['analysis_type'] = analysis_type

# Get active indicators with unique keys
active_indicators = sidebar_indicator_selection(strategy_type, interval)


# Fetch Data Button with Enhanced Logic
# This button fetches data based on user inputs and updates the session state
if st.sidebar.button("🔄 Fetch & Analyze Data", type="primary"):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        status_text.text("📈 Fetching market data...")
        progress_bar.progress(25)
        
        data, levels, options_data = fetch_and_process_data(
            ticker, start_date, end_date, interval, strategy_type, analysis_type, 
            active_indicators
        )
        progress_bar.progress(75)
        
        if data is not None:
            status_text.text("🔧 Calculating indicators...")
            st.session_state["stock_data"] = data
            st.session_state["levels"] = levels
            st.session_state["options_data"] = options_data
            st.session_state["active_indicators"] = active_indicators
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Show summary in sidebar
            st.sidebar.success(f"✅ Loaded {len(data)} {interval} candles for {ticker}")
            
            # Clear progress after 2 seconds
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        else:
            progress_bar.empty()
            status_text.empty()
            st.sidebar.error("❌ Failed to fetch data. Check ticker symbol and date range.")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.sidebar.error(f"❌ Error: {str(e)}")
        logging.error(f"Data fetch error: {str(e)}")

# --- MAIN ANALYSIS SECTION ---
# This section displays the stock data, technical indicators, and analysis results
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]
    
    # Make sure levels is a dictionary with 'support' and 'resistance' keys
    if "levels" in st.session_state and isinstance(st.session_state["levels"], dict):
        levels = st.session_state["levels"]
    else:
        levels = {'support': [], 'resistance': []}
        
    options_data = st.session_state.get("options_data", {})
    ticker_str = ticker.upper()
    
    # Create tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["📈 Technical Analysis", "🤖 AI Recommendation", "💰 Options Analyzer"])
    
    # --- TAB 1: TECHNICAL ANALYSIS ---
    with tab1:
        # Show key metrics at the top
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



    # Get fundamentals with better error handling
    try:
        with st.spinner("Fetching fundamental data..."):
            fundamentals = get_fundamental_metrics(ticker)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch fundamental data. Using basic metrics only.")
        fundamentals = {}

    st.markdown("### 📊 Stock Metrics")
    metric_labels = [
        ("IV Rank", f"{options_data['iv_data'].get('iv_rank', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("IV Percentile", f"{options_data['iv_data'].get('iv_percentile', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("30-Day HV", f"{options_data['iv_data'].get('hv_30', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("VIX Level", f"{options_data['iv_data'].get('vix', 0):.1f}" if options_data and options_data.get("iv_data") else "N/A"),
        ("EPS", f"{fundamentals.get('EPS', 0):.2f}" if isinstance(fundamentals.get('EPS'), (int, float)) else "N/A"),
        ("P/E Ratio", f"{fundamentals.get('P/E Ratio', 0):.2f}" if isinstance(fundamentals.get('P/E Ratio'), (int, float)) else "N/A"),
        ("Revenue Growth", f"{fundamentals.get('Revenue Growth', 0):.1f}%" if isinstance(fundamentals.get('Revenue Growth'), (int, float)) else "N/A"),
        ("Profit Margin", f"{fundamentals.get('Profit Margin', 0):.1f}%" if isinstance(fundamentals.get('Profit Margin'), (int, float)) else "N/A"),
    ]

    cols = st.columns(4)
    for i, (label, value) in enumerate(metric_labels):
        with cols[i % 4]:
            st.metric(label, value if value is not None else "N/A")


    # Enhanced Chart with Strategy-Specific Indicators
    st.markdown("### 📈 Technical Analysis Chart")

    # Optional: Show current indicator values in expandable section
    with st.expander("📊 Current Indicator Values", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        latest = data.iloc[-1]
        
        with col1:
            st.markdown("**Trend Indicators**")
            if 'SMA_20' in data.columns and pd.notna(latest['SMA_20']):
                st.metric("SMA(20)", f"${latest['SMA_20']:.2f}")
            if 'EMA_20' in data.columns and pd.notna(latest['EMA_20']):
                st.metric("EMA(20)", f"${latest['EMA_20']:.2f}")
            if 'VWAP' in data.columns and pd.notna(latest['VWAP']):
                st.metric("VWAP", f"${latest['VWAP']:.2f}")
        
        with col2:
            st.markdown("**Momentum Indicators**")
            if 'RSI' in data.columns and pd.notna(latest['RSI']):
                rsi_color = "🔴" if latest['RSI'] > 70 else "🟢" if latest['RSI'] < 30 else "🟡"
                st.metric("RSI(14)", f"{latest['RSI']:.1f} {rsi_color}")
            if 'MACD' in data.columns and pd.notna(latest['MACD']):
                st.metric("MACD", f"{latest['MACD']:.4f}")
            if 'ADX' in data.columns and pd.notna(latest['ADX']):
                st.metric("ADX(14)", f"{latest['ADX']:.1f}")
        
        with col3:
            st.markdown("**Volatility & Volume**")
            if 'ATR' in data.columns and pd.notna(latest['ATR']):
                st.metric("ATR(14)", f"${latest['ATR']:.2f}")
            if 'OBV' in data.columns and pd.notna(latest['OBV']):
                st.metric("OBV", f"{latest['OBV']:,.0f}")
            if 'volatility' in data.columns and pd.notna(latest['volatility']):
                st.metric("Historical Vol", f"{latest['volatility']*100:.1f}%")

    subplot_fig, daily_fig, timeframe_fig = plotter.create_enhanced_chart(
        data=data,
        indicators=st.session_state["active_indicators"],
        levels=levels,
        strategy_type=strategy_type,
        options_data=options_data,
        interval=interval
    )

    # Print available columns to the console for debugging
    # print("Available columns:", list(data.columns))
    # ...existing code...
    
    # Log indicator summary 
    base_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    indicator_columns = [col for col in data.columns if col not in base_columns]
    
    logging.info(f"📊 Dashboard loaded: {len(indicator_columns)} technical indicators calculated")
    logging.debug(f"🔍 Indicator list: {', '.join(sorted(indicator_columns))}")


    st.plotly_chart(subplot_fig, use_container_width=True, height=1000)
    
    # --- Display interactive chart in Tab 1 ---
    with tab1:
        # Display interactive chart with plotly
        try:
            # Ensure levels is a properly formatted dictionary
            if not isinstance(levels, dict) or not ("support" in levels and "resistance" in levels):
                levels = {'support': [], 'resistance': []}
                
            subplot_fig, daily_fig, timeframe_fig = plotter.create_enhanced_chart(
                data=data,
                indicators=st.session_state["active_indicators"], 
                levels=levels,
                interval=interval
            )
        except Exception as chart_error:
            st.error(f"Error creating chart: {str(chart_error)}")
            # Create a simple fallback chart
            import plotly.graph_objects as go
            subplot_fig = go.Figure()
            daily_fig = go.Figure() 
            timeframe_fig = go.Figure()
            
            # Add data to all charts
            for chart in [subplot_fig, daily_fig, timeframe_fig]:
                chart.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
        st.plotly_chart(subplot_fig, use_container_width=True)
        
        # Display technical indicators summary
        st.markdown("### Technical Indicators Summary")
        
        # Group indicators by type
        indicator_groups = {
            "Trend Indicators": ["SMA", "EMA", "MACD", "ADX"],
            "Momentum Indicators": ["RSI", "STOCH", "CCI"],
            "Volatility Indicators": ["BBands", "ATR", "Standard Deviation"],
            "Volume Indicators": ["OBV", "VWAP", "Volume"]
        }
        
        # Create columns for indicator categories
        ind_cols = st.columns(len(indicator_groups))
        
        # Display indicator values if available
        for i, (group_name, indicators) in enumerate(indicator_groups.items()):
            with ind_cols[i]:
                st.markdown(f"**{group_name}**")
                for ind in indicators:
                    for col in data.columns:
                        if ind in col:
                            try:
                                value = data[col].iloc[-1]
                                if isinstance(value, (int, float)):
                                    st.metric(col, f"{value:.2f}")
                            except:
                                pass
    
    # --- TAB 2: AI ANALYSIS ---
    with tab2:
        st.markdown("### 🤖 AI-Powered Strategy Analysis")
        
        # Check if we already have analysis results
        if "ai_analysis_result" in st.session_state:
            analysis, chart_path, recommendation = st.session_state["ai_analysis_result"]
            
            # Convert the analysis to a nicely formatted version for display
            formatted_analysis = format_analysis_text(analysis)
            
            # Display headline recommendation
            if recommendation and 'action' in recommendation:
                action = recommendation['action']
                confidence = recommendation.get('confidence', 0) * 100
                strategy_name = recommendation.get('strategy', {}).get('name', 'N/A')
                
                if action == 'BUY':
                    action_color = "🟢"
                    action_style = "success"
                elif action == 'SELL':
                    action_color = "🔴"
                    action_style = "error"
                else:
                    action_color = "🟡"
                    action_style = "info"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #1f77b4;
                    margin: 10px 0;
                ">
                    <h4 style="margin: 0; color: #1f77b4;">📋 Analysis Summary</h4>
                    <p style="margin: 5px 0; font-size: 18px;">
                        <strong>Recommendation:</strong> {action_color} <strong>{action}</strong> 
                        | <strong>Strategy:</strong> {strategy_name} 
                        | <strong>Confidence:</strong> {confidence:.0f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display the full analysis
                st.markdown(formatted_analysis)
        else:
            # No analysis yet, show the Run Analysis button
            analysis_cols = st.columns([1])
            
            with analysis_cols[0]:
                run_analysis = st.button("Run Analysis 💸", type="primary", use_container_width=True)
                if run_analysis:
                    st.session_state['run_analysis'] = True
            
    # --- TAB 3: OPTIONS ANALYZER ---
    with tab3:
        if analysis_type == "Options Trading Strategy":
            try:
                # Import and use our options analyzer component
                from src.ui_components.options_analyzer import display_options_analyzer
                
                # Display options analyzer component
                display_options_analyzer(ticker_str, data, options_data.get(ticker_str, {}))
            except ImportError as e:
                st.error(f"Error loading Options Analyzer: {str(e)}")
                st.info("Please ensure the options analyzer component is properly installed.")
        else:
            st.info("Select 'Options Trading Strategy' in the sidebar to use the Options Analyzer.")
            
            # Show a quick options trading overview
            st.subheader("Options Trading Overview")
            st.markdown("""
            Options trading provides flexible strategies for various market conditions:
            
            - **Bullish strategies**: Long calls, bull call spreads, bull put spreads
            - **Bearish strategies**: Long puts, bear put spreads, bear call spreads
            - **Neutral strategies**: Iron condors, butterflies, calendar spreads
            - **Volatility strategies**: Straddles, strangles
            
            Switch to 'Options Trading Strategy' in the sidebar to access the full Options Analyzer.
            """)

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

    # Add strategy context
    if options_strategy:
        selected_strategy_info = get_strategy_by_name(options_strategy)
        if selected_strategy_info:
            # Get first available timeframe for strategy context
            timeframes = selected_strategy_info.get('Timeframes', {})
            first_timeframe = list(timeframes.keys())[0] if timeframes else None
            
            if first_timeframe:
                timeframe_data = timeframes[first_timeframe]
                strategy_context = f"""
                SELECTED STRATEGY CONTEXT:
                - Strategy: {selected_strategy_info['Strategy']}
                - Timeframe: {first_timeframe}
                - Best Use: {timeframe_data.get('Best_Use', 'N/A')}
                - Key Indicators: {', '.join(timeframe_data.get('Key_Indicators', [])[:3])}
                - Advanced Tips: {', '.join(timeframe_data.get('Advanced_Tips', [])[:2])}
                """
                market_context += "\n" + strategy_context

    if options_data:
        market_context += f"""
        OPTIONS MARKET DATA:
        - IV Rank: {options_data.get('iv_data', {}).get('iv_rank', 'N/A')}%
        - IV Percentile: {options_data.get('iv_data', {}).get('iv_percentile', 'N/A')}%
        - 30-Day Historical Volatility: {options_data.get('iv_data', {}).get('hv_30', 'N/A')}%
        """

    # Strategy-specific prompts
    if analysis_type == "Stock Buy/Hold/Sell":
        if "Short-Term" in strategy_type:
            prompt = f"""
            You are an expert short-term stock trader. Analyze this {interval} chart for {ticker}.
            
            {market_context}
            
            FOCUS ON SHORT-TERM INDICATORS (1-7 days):
            - RSI signals and momentum
            - MACD crossovers and trends
            - Volume patterns and breakouts
            - Short-term moving averages
            - Price action and candlestick patterns
            - Support/resistance levels
            
            PROVIDE:
            1. **RECOMMENDATION**: [BUY/SELL/HOLD]
            2. **ENTRY POINTS**: Specific price levels
            3. **STOP LOSS**: Based on support levels and ATR
            4. **PROFIT TARGETS**: Multiple take-profit levels
            5. **KEY INDICATORS**: Most relevant signals
            6. **RISK/REWARD**: Ratio and position sizing
            """
        else:  # Long-term
            prompt = f"""
            You are an expert long-term stock analyst. Analyze this {interval} chart for {ticker}.
            
            {market_context}
            
            FOCUS ON LONG-TERM INDICATORS:
            - Trend strength and direction
            - Moving average crossovers
            - Volume trends and accumulation
            - Long-term support/resistance
            - Market sentiment indicators
            
            PROVIDE:
            1. **RECOMMENDATION**: [BUY/SELL/HOLD]
            2. **TIMEFRAME**: Expected holding period
            3. **ENTRY STRATEGY**: Buy zones and conditions
            4. **RISK MANAGEMENT**: Stop loss levels
            5. **TARGET PRICES**: Based on technical levels
            6. **TREND ANALYSIS**: Primary and secondary trends
            """
    elif analysis_type == "Options Trading Strategy":
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
            1. **TRADE RECOMMENDATION**: [YES/NO]
            2. **STRATEGY**: Specific options strategy ({options_strategy})
            3. **STRIKES & EXPIRATION**: Based on ATR and support/resistance
            4. **ENTRY/EXIT CRITERIA**: Specific indicator levels
            5. **RISK MANAGEMENT**: Stop loss and profit targets
            6. **RATIONALE**: Why this setup works for short-term options
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
            1. **TRADE RECOMMENDATION**: [YES/NO]
            2. **STRATEGY**: Specific options strategy ({options_strategy})
            3. **STRIKES & EXPIRATION**: 2-3 weeks out, based on major levels
            4. **TREND ANALYSIS**: Primary trend direction and strength
            5. **RISK/REWARD**: Expected profit targets and stop losses
            6. **RATIONALE**: Why this setup works for swing trading
            """

    # Run AI analysis synchronously (for simplicity)

    if st.session_state.get('run_analysis', False):
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🤖 AI is analyzing the market..."):
            print("\n" + "="*60)
            print("🤖 STARTING AI MARKET ANALYSIS")
            print("="*60)
            
            # Step 1: Price Prediction (20%)
            status_text.text("🔮 Generating price predictions...")
            progress_bar.progress(20)
            
            # Get the price prediction using ensemble of models
            try:
                prediction_result = predict_next_day_close(
                    data.copy(),
                    fundamentals,
                    st.session_state["active_indicators"]
                )
                
                if prediction_result and isinstance(prediction_result, tuple) and prediction_result[0] is not None:
                    predicted_price, confidence = prediction_result
                    # Create a new column filled with the predicted price
                    data['Predicted_Close'] = float(predicted_price)
                    price_change = predicted_price - data['Close'].iloc[-1]
                    data['Predicted_Price_Change'] = price_change
                    # Update market context with predicted price and confidence
                    market_context += f"\nNEXT DAY PREDICTED CLOSE: ${predicted_price:.2f} (Change: ${price_change:.2f}, Confidence: {confidence:.1%})\n"

                    # Update the prompt with prediction info
                    prediction_context = f"""PREDICTED NEXT DAY CLOSE: ${predicted_price:.2f} (Confidence: {confidence:.1%})
PRICE CHANGE: ${price_change:.2f} ({(price_change/data['Close'].iloc[-1]*100):.1f}%)"""
                    print(f"✅ Price prediction: ${predicted_price:.2f} (Confidence: {confidence:.1%})")
                else:
                    print("⚠️ AI price prediction temporarily unavailable (insufficient data)")
                    data_size = len(data)
                    print(f"📊 Current dataset: {data_size} rows (minimum 20 recommended)")
                    prediction_context = f"AI PRICE PREDICTION: Unavailable due to small dataset ({data_size} rows). Consider using a longer date range."
            except Exception as e:
                print(f"⚠️ Prediction error: {str(e)}")
                prediction_context = "AI PRICE PREDICTION: Temporarily unavailable"

            # Step 2: Prepare Chart (40%)
            status_text.text("📊 Preparing chart analysis...")
            progress_bar.progress(40)

            if "Short-Term" in strategy_type:
                prompt = prompt.replace("PROVIDE:", f"{prediction_context}\nCONSIDER HOW THIS PRICE AFFECTS SHORT-TERM INDICATORS AND MOMENTUM.\n\nPROVIDE:")
            else:
                prompt = prompt.replace("PROVIDE:", f"{prediction_context}\nCONSIDER HOW THIS PRICE AFFECTS TRENDS AND LONG-TERM STRATEGIES.\n\nPROVIDE:")

            # Create temporary chart file using temp manager
            chart_path = temp_manager.create_chart_file(ticker)
            
            # Save chart to temporary file (suppress verbose logging)
            import logging as base_logging
            kaleido_logger = base_logging.getLogger('kaleido')
            original_level = kaleido_logger.level
            kaleido_logger.setLevel(base_logging.WARNING)
            
            try:
                subplot_fig.write_image(chart_path)
                print(f"✅ Chart prepared for AI analysis")
            except Exception as e:
                print(f"❌ Error saving chart: {e}")
                st.error(f"Failed to save chart: {e}")
            finally:
                kaleido_logger.setLevel(original_level)
            
            # Step 3: Run AI Analysis (60%)
            status_text.text("🧠 Running AI analysis...")
            progress_bar.progress(60)
            
            # Run AI analysis with user-configured settings
            try:
                # Use user-defined timeout, with strategy-based adjustment
                if "Short-Term" in strategy_type:
                    adjusted_timeout = max(vision_timeout - 30, 30)  # Reduce for short-term
                else:
                    adjusted_timeout = vision_timeout
                
                if enable_vision_analysis:
                    status_text.text(f"🧠 Running AI analysis (Vision timeout: {adjusted_timeout}s)...")
                else:
                    status_text.text("🧠 Running AI analysis (Vision analysis disabled)...")
                    adjusted_timeout = 0  # Skip vision analysis
                
                analysis, recommendation = run_ai_analysis(
                    daily_fig=daily_fig,
                    timeframe_fig=timeframe_fig,
                    data=data,
                    ticker=ticker,
                    prompt=prompt,
                    vision_timeout=adjusted_timeout,
                    options_priority=options_priority
                )
                
                # Step 4: Complete Analysis (100%)
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                print("✅ AI MARKET ANALYSIS COMPLETED")
                print("="*60 + "\n")
                
                st.session_state["ai_analysis_result"] = (analysis, chart_path, recommendation)
                # Reset the run_analysis flag to prevent re-running on page refresh
                st.session_state['run_analysis'] = False
            except Exception as e:
                status_text.text("❌ Analysis failed")
                progress_bar.progress(0)
                st.error(f"AI analysis failed: {e}")
                import traceback
                traceback.print_exc()
                # Reset the run_analysis flag even if there was an error
                st.session_state['run_analysis'] = False
            finally:
                # Clear progress indicators after a short delay
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

    if st.session_state.get("ai_analysis_result") is None and st.session_state.get("ai_analysis_running"):
        st.info("AI analysis started... Please wait.")
        st.spinner("AI is analyzing the market...")

    if st.session_state.get("ai_analysis_result"):
        analysis, chart_path, recommendation = st.session_state["ai_analysis_result"]
        
        # Display Analysis Results with Enhanced Formatting
        st.markdown("### 🤖 AI Trading Analysis Results")
        st.markdown("---")  # Add a separator line
        
        # Quick Summary Card
        if recommendation:
            action = recommendation.get('action', 'Hold').upper()
            strategy_name = recommendation.get('strategy', {}).get('name', 'No Strategy')
            confidence = recommendation.get('strategy', {}).get('confidence', 0) * 100
            
            # Color code the action
            if action == 'BUY':
                action_color = "🟢"
                action_style = "success"
            elif action == 'SELL':
                action_color = "🔴"
                action_style = "error"
            else:
                action_color = "🟡"
                action_style = "info"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
                margin: 10px 0;
            ">
                <h4 style="margin: 0; color: #1f77b4;">📋 Analysis Summary</h4>
                <p style="margin: 5px 0; font-size: 18px;">
                    <strong>Recommendation:</strong> {action_color} <strong>{action}</strong> 
                    | <strong>Strategy:</strong> {strategy_name} 
                    | <strong>Confidence:</strong> {confidence:.0f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategy Overview with better styling
        if recommendation and 'strategy' in recommendation:
            strategy = recommendation['strategy']
            
            st.markdown("#### 🎯 Strategy Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                strategy_name = strategy.get('name', 'N/A')
                st.metric("Strategy", strategy_name, help="Recommended trading strategy")
            with col2:
                confidence = strategy.get('confidence', 0) * 100
                confidence_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
                st.metric("Confidence", f"{confidence:.0f}%", help="AI confidence level")
            with col3:
                # Fix: get risk_level from risk_assessment, not strategy
                risk_level = recommendation.get('risk_assessment', {}).get('risk_level', 'N/A')
                risk_color = "🔴" if str(risk_level).upper() == "HIGH" else "🟡" if str(risk_level).upper() == "MEDIUM" else "🟢"
                st.metric("Risk Level", f"{risk_color} {str(risk_level).upper()}", help="Assessed risk level")
        
        # Market Analysis Metrics with improved presentation
        if recommendation and 'market_analysis' in recommendation:
            market = recommendation['market_analysis']
            st.markdown("#### 📊 Current Market Conditions")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                rsi_val = market.get('RSI', 0)
                rsi_signal = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Neutral"
                st.metric("RSI", f"{rsi_val:.1f}", help=f"Relative Strength Index: {rsi_signal}")
            with metrics_col2:
                macd_signal = market.get('MACD_Signal', 'N/A')
                macd_emoji = "🟢" if macd_signal == "bullish" else "🔴" if macd_signal == "bearish" else "🟡"
                st.metric("MACD", f"{macd_emoji} {str(macd_signal).title()}", help="MACD trend signal")
            with metrics_col3:
                volume_signal = market.get('volume_signal', 'N/A')
                volume_emoji = "🟢" if volume_signal == "high" else "🔴" if volume_signal == "low" else "🟡"
                st.metric("Volume", f"{volume_emoji} {str(volume_signal).title()}", help="Trading volume analysis")
            with metrics_col4:
                trend_strength = market.get('trend_strength', 0)
                trend_signal = "💪 Strong" if trend_strength > 25 else "📈 Weak" if trend_strength > 15 else "➡️ Sideways"
                st.metric("Trend (ADX)", f"{trend_strength:.1f}", help=f"Trend strength: {trend_signal}")
        
        # Trade Parameters - Format in a more readable way
        if recommendation and 'parameters' in recommendation and recommendation['parameters']:
            st.markdown("#### 📈 Trade Parameters")
            params = recommendation['parameters']
            
            # Create formatted parameter display
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                if 'entry_condition' in params and params['entry_condition'] is not None:
                    st.info(f"**Entry Condition:** {params['entry_condition'].replace('_', ' ').title()}")
                elif 'entry_condition' in params:
                    st.info("**Entry Condition:** Not specified")
                if 'stop_loss' in params:
                    try:
                        stop_loss_val = float(params['stop_loss'])
                        st.error(f"**Stop Loss:** ${stop_loss_val:.2f}")
                    except (ValueError, TypeError):
                        st.error(f"**Stop Loss:** {params['stop_loss']}")
                        
            with param_col2:
                if 'exit_condition' in params and params['exit_condition'] is not None:
                    st.info(f"**Exit Condition:** {params['exit_condition'].replace('_', ' ').title()}")
                elif 'exit_condition' in params:
                    st.info("**Exit Condition:** Not specified")
                if 'profit_target' in params:
                    try:
                        profit_val = float(params['profit_target'])
                        st.success(f"**Profit Target:** ${profit_val:.2f}")
                    except (ValueError, TypeError):
                        st.success(f"**Profit Target:** {params['profit_target']}")
                        
            # Additional parameters in a clean format
            other_params = {k: v for k, v in params.items() 
                          if k not in ['entry_condition', 'exit_condition', 'stop_loss', 'profit_target']}
            
            if other_params:
                st.markdown("**Additional Parameters:**")
                for key, value in other_params.items():
                    formatted_key = key.replace('_', ' ').title()
                    if isinstance(value, bool):
                        value_display = "✅ Yes" if value else "❌ No"
                    elif isinstance(value, (int, float)):
                        if 'period' in key.lower() or 'ma' in key.lower():
                            value_display = f"{value} periods"
                        else:
                            value_display = f"{value:.2f}"
                    else:
                        if value is not None:
                            value_display = str(value).replace('_', ' ').title()
                        else:
                            value_display = "Not specified"
                    
                    st.write(f"• **{formatted_key}:** {value_display}")
        
        # Professional Analysis Display
        st.markdown("#### 📝 Professional Trade Report")
        
        # Use the new professional formatting function
        try:
            professional_report = format_professional_report(
                analysis, recommendation, ticker, strategy_type, options_strategy, 
                data, levels, options_data
            )
            
            # Display the professional report
            st.markdown(professional_report)
            
        except Exception as format_error:
            st.warning("⚠️ Error formatting professional report. Showing standard format.")
            # Fallback to original formatting
            if analysis:
                cleaned_analysis = format_analysis_text(analysis)
                st.markdown(cleaned_analysis)
            else:
                st.info("📝 No detailed analysis available. Please run the analysis to get AI insights.")
        
        # Action Buttons Section
        st.markdown("---")
        st.markdown("#### 📋 Report & Actions")
        
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            # Enhanced PDF Generation
            if st.button("📄 Generate Detailed Report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    try:
                        # Generate professional report for PDF
                        professional_report = format_professional_report(
                            analysis, recommendation, ticker, strategy_type, options_strategy, 
                            data, levels, options_data
                        )
                        
                        generate_and_display_pdf(
                            ticker, strategy_type, options_strategy, data, professional_report, chart_path, levels, options_data, st.session_state["active_indicators"]
                        )
                        st.success("✅ PDF report generated successfully!")
                        print("✅ PDF report generated successfully")
                        
                        # Clean up temporary chart file after PDF generation
                        if chart_path and os.path.exists(chart_path):
                            temp_manager.cleanup_file(chart_path)
                            print(f"🗑️ Cleaned up temporary chart: {chart_path}")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {e}")
                        print(f"❌ PDF generation error: {e}")
        
        with button_col2:
            # Clean up temp files when user closes analysis
            if st.button("🗑️ Clear Analysis & Clean Temp Files", use_container_width=True):
                if "ai_analysis_result" in st.session_state:
                    del st.session_state["ai_analysis_result"]
                temp_manager.cleanup_all()
                st.success("✅ Analysis cleared and temporary files cleaned up!")
                st.rerun()

        st.session_state["ai_analysis_running"] = False


# --- SIDEBAR: QUICK STATS ---
if "stock_data" in st.session_state:
    render_sidebar_quick_stats(st.session_state["stock_data"], interval)


# Footer
st.markdown("---")
st.markdown("*This analysis is for educational purposes only. Always conduct your own research before trading.*")