import streamlit as st
import pandas as pd
import os
import logging 
from src import plotter, ai_analysis, config
from src.data_pipeline import fetch_and_process_data
from src.prediction import get_fundamental_metrics, predict_next_day_close
from src.pdf_utils import generate_and_display_pdf
from src.ui_components import render_sidebar_quick_stats, sidebar_config, sidebar_indicator_selection
from src.trading_strategies import strategies_data  # Add this import at the top
from src.logging_config import setup_logging, set_log_level
from src.temp_manager import temp_manager, cleanup_old_temp_files

# Setup cleaner logging for Streamlit
setup_logging(level=logging.INFO, enable_file_logging=False)

# Suppress verbose libraries
logging.getLogger('kaleido').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Clean up old temp files on startup
cleanup_old_temp_files()

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

# --- Model Selection Widget ---
model_type = st.sidebar.selectbox(
    "Select Model",
    ["RandomForest", "XGBoost", "CatBoost"],
    index=0,
    help="Choose which machine learning model to use for price prediction."
)

# --- Modular Sidebar ---
# Stock ticker, date range, timeframee/interval, analysis type, strategy type, technical indicators
ticker, start_date, end_date, interval, analysis_type, strategy_type, options_strategy = sidebar_config(config)
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

    fig = plotter.create_enhanced_chart(
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


    st.plotly_chart(fig, use_container_width=True, height=800)
    
    # --- ENHANCED AI ANALYSIS ---
    st.markdown("### 🤖 AI-Powered Strategy Analysis")
    
    analysis_cols = st.columns([1])  # Changed from [2, 1] to just [1]
    
    with analysis_cols[0]:
        run_analysis = st.button("Run Analysis 💸", type="primary", use_container_width=True)

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
        selected_strategy_info = next((s for s in strategies_data if s["Strategy"] == options_strategy), None)
        if selected_strategy_info:
            strategy_context = f"""
            SELECTED STRATEGY CONTEXT:
            - Strategy: {selected_strategy_info['Strategy']}
            - Description: {selected_strategy_info['Description']}
            - Timeframe: {selected_strategy_info['Timeframe']}
            - Pros: {', '.join(selected_strategy_info['Pros'])}
            - Cons: {', '.join(selected_strategy_info['Cons'])}
            - When to Use: {selected_strategy_info['When to Use']}
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

    if run_analysis:
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
            
            # Get the price prediction with selected model
            try:
                prediction_result = predict_next_day_close(
                    data.copy(),
                    fundamentals,
                    st.session_state["active_indicators"],
                    model_type=model_type
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
                fig.write_image(chart_path)
                print(f"✅ Chart prepared for AI analysis")
            except Exception as e:
                print(f"❌ Error saving chart: {e}")
                st.error(f"Failed to save chart: {e}")
            finally:
                kaleido_logger.setLevel(original_level)
            
            # Step 3: Run AI Analysis (60%)
            status_text.text("🧠 Running AI analysis...")
            progress_bar.progress(60)
            
            # Run AI analysis
            try:
                analysis, recommendation = ai_analysis.run_ai_analysis(
                    fig=fig,
                    data=data,
                    ticker=ticker,
                    prompt=prompt
                )
                
                # Step 4: Complete Analysis (100%)
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                print("✅ AI MARKET ANALYSIS COMPLETED")
                print("="*60 + "\n")
                
                st.session_state["ai_analysis_result"] = (analysis, chart_path, recommendation)
            except Exception as e:
                status_text.text("❌ Analysis failed")
                progress_bar.progress(0)
                st.error(f"AI analysis failed: {e}")
                import traceback
                traceback.print_exc()
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
        
        # Display Analysis Results
        st.markdown("### 📋 AI Trading Analysis")
        
        # Strategy Overview
        if recommendation and 'strategy' in recommendation:
            strategy = recommendation['strategy']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strategy", strategy.get('name', 'N/A'))
            with col2:
                st.metric("Confidence", f"{strategy.get('confidence', 0) * 100:.0f}%")
            with col3:
                # Fix: get risk_level from risk_assessment, not strategy
                risk_level = recommendation.get('risk_assessment', {}).get('risk_level', 'N/A')
                st.metric("Risk Level", str(risk_level).upper())
        
        # Market Analysis Metrics
        if recommendation and 'market_analysis' in recommendation:
            market = recommendation['market_analysis']
            st.markdown("#### 📊 Market Conditions")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("RSI", f"{market.get('RSI', 0):.2f}")
            with metrics_col2:
                st.metric("MACD Signal", market.get('MACD_Signal', 'N/A'))
            with metrics_col3:
                st.metric("Volume", market.get('volume_signal', 'N/A'))
            with metrics_col4:
                st.metric("Trend Strength", f"{market.get('trend_strength', 0):.2f}")
        
        # Trade Parameters
        if recommendation and 'parameters' in recommendation and recommendation['parameters']:
            st.markdown("#### 📈 Trade Parameters")
            st.json(recommendation['parameters'])
        
        # Full Analysis
        st.markdown("#### � Detailed Analysis")
        st.markdown(analysis)
        # Enhanced PDF Generation
        if st.button("📄 Generate Detailed Report"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    generate_and_display_pdf(
                        ticker, strategy_type, options_strategy, data, analysis, chart_path, levels, options_data, st.session_state["active_indicators"]
                    )
                    print("✅ PDF report generated successfully")
                    
                    # Clean up temporary chart file after PDF generation
                    if chart_path and os.path.exists(chart_path):
                        temp_manager.cleanup_file(chart_path)
                        print(f"🗑️ Cleaned up temporary chart: {chart_path}")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
                    print(f"❌ PDF generation error: {e}")

        st.session_state["ai_analysis_running"] = False

        # Optional: Clean up temp files when user closes analysis
        if st.button("🗑️ Clear Analysis & Clean Temp Files"):
            if "ai_analysis_result" in st.session_state:
                del st.session_state["ai_analysis_result"]
            temp_manager.cleanup_all()
            st.success("Analysis cleared and temporary files cleaned up!")
            st.rerun()



# --- SIDEBAR: QUICK STATS ---
if "stock_data" in st.session_state:
    render_sidebar_quick_stats(st.session_state["stock_data"], interval)


# Footer
st.markdown("---")
st.markdown("*This analysis is for educational purposes only. Always conduct your own research before trading.*")