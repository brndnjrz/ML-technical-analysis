import streamlit as st
import pandas as pd
from src import config, data_loader, indicators, plotter, ai_analysis, pdf_generator
import tempfile
import base64
import os

# Set Up Streamlit App UI 
st.set_page_config(page_title="AI Technical Analysis", layout="wide")   # Page layout set to full width
st.title("Technical Stock Analysis Dashboard")  # Displays the title at the top
st.sidebar.header("Configuration")  # Header in the sidebar

# --- Sidebar Input --- 
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", config.DEFAULT_TICKER)
start_date = st.sidebar.date_input("Start Date", value=config.DEFAULT_START_DATE)
end_date = st.sidebar.date_input("End Date", value=config.DEFAULT_END_DATE)

# Add interval selection 
interval = st.sidebar.selectbox("Select Candle Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

# RSI, MACD, and New Indicator Toggles
show_rsi = st.sidebar.toggle("Show RSI", value=True)
show_macd = st.sidebar.toggle("Show MACD", value=True)
show_adx = st.sidebar.toggle("Show ADX", value=False)
show_stoch = st.sidebar.toggle("Show Stochastic Oscillator", value=False)
show_obv = st.sidebar.toggle("Show OBV", value=False)
show_atr = st.sidebar.toggle("Show ATR", value=False)


# Strategy Selection 
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type:", 
    ["Options Trading Strategy", "Buy/Hold/Sell Recommendation"])

# If Options Strategy is selected, provide further options 
strategy_type = None
if analysis_type == "Options Trading Strategy":
    strategy_type = st.sidebar.selectbox(
        "Options Trading Strategy:", 
        ["Short-Term", "Long-Term"], index=1)

# Fetch and Store Stock Data 
if st.sidebar.button("Fetch Data"):
    data = data_loader.fetch_stock_data(ticker, start_date, end_date, interval)
    if data is None or data.empty:
        st.warning("No data found for selected interval and dates.")
        st.stop()
    else:
        st.session_state["stock_data"] = data
        st.success("Stock data loaded successfully!")

# --- Main Analysis --- 
if "stock_data" in st.session_state:
    print("Debug: Columns in stock_data ->", st.session_state["stock_data"].columns.tolist())
    print("Debug: Head of stock_data ->\n", st.session_state["stock_data"].head())


    data = indicators.calculate_indicators(st.session_state["stock_data"], timeframe=interval)
    # Detect support and resistance zones
    levels = indicators.detect_support_resistance(data)

    # Support and Resistance 
    st.markdown("### Support & Resistance Levels")
    st.write(f"**Support Zones:** {['${:.2f}'.format(l) for l in levels['support']]}")
    st.write(f"**Resistance Zones:** {['${:.2f}'.format(l) for l in levels['resistance']]}")

    # Sidebar: Select Technical Indicators
    selected_indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "50-Day SMA", "20-Day EMA", "50-Day EMA", "Implied Volatility", "Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )
    # Final Chart Formatting and Display 
    fig = plotter.create_chart(
        data=data, 
        indicators=selected_indicators, 
        show_rsi=show_rsi, 
        show_macd=show_macd, 
        levels=levels, 
        show_adx=show_adx, 
        show_stoch=show_stoch, 
        show_obv=show_obv, 
        show_atr=show_atr)
    # Show active indicators above the chart
    st.write(f"**Showing indicators based on `{interval}` candles**")
    st.plotly_chart(fig, use_container_width=True, height=1200)

    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing the chart, please wait..."):
            sr_text = f"""
                Current Support Levels: {['${:.2f}'.format(l) for l in levels['support']]}
                Current Resistance Levels: {['${:.2f}'.format(l) for l in levels['resistance']]}
                """
            # Choose prompt based on analysis type and strategy type
            if analysis_type == "Options Trading Strategy":
                if strategy_type == "Short-Term":
                    prompt = f"""
                    Prompt:
                        You are a professional Options Trader and Technical Analyst. Focus on short-term options trading strategies (1 to 7 days) based on intraday technical indicators and price action.                         

                    Instructions:
                        1. Recommend a short-term options strategy (e.g., Iron Condor, intraday straddle, protective put, covered call, quick spreads) based on:
                            - Candlestick patterns and intraday support/resistance
                            - 5m/15m/1h/1d RSI, MACD, ADX, ATR, Stochastic Oscillator, OBV
                            - SMA/EMA, Bollinger Bands, VWAP
                            - Volume spikes and Implied Volatility shifts 
                        2. Start by stating:
                            - Recommended options strategy
                            - Whether to enter the trade (Yes or No), based only on technical evidence.
                        3. Then , explain:
                            - Why the selected strategy fits the current technical setup.
                            - Suggested short-term strick prices and expiration date (3-7 days), using current price action.
                            - Risk/reward profile: potential gain vs. max loss, ideal entry/exit points
                            - Technical indicators that support the strategy
                    Guidelines:
                        - Use short-term logic, e.g., 5m/15m/1h/1d levels, fast-moving indicators
                        - Be concise, professional, and evidence-based.
                        - Do not recommend a trade unless indicators show clear confirmation
                        """
                else:
                    prompt = f"""
                    Prompt:
                        You are a professional Options Trader and Technical Analyst. Focus on long-term swing trading strategies (1 to 3 weeks) based on daily/weekly chart technical indicators. 
                    
                    

                    Instructions:
                        1. Recommend a long-term options strategy (e.g., Iron Condor, Vertical Spread, Covered Call, Protective Put) using: 
                            - Daily/weekly SMA/EMA, RSI, MACD, ADX
                            - Bollinger Bands, ATR, OBV, Implied Volatility (IV), and Volume
                            - Stochastic Oscillator and support/resistance zones
                        2. Start by stating:
                            - Recommended options strategy (e.g., Iron Condor, Call Spread, Put Spread, Covered Call, Protective Put)
                            - Whether to enter the trade (Yes or No), based only on technical evidence.
                        3. Then , explain:
                            - Why the selected strategy fits the current technical setup.
                            - Suggested strick prices and expiration date(1-3 weeks)
                            - Risk/reward profile: potential gain vs. max loss
                            - Ideal entry/exit levels based on technical indicators
                    Guidelines:
                        - Use a structured, evidence-based approach
                        - Be concise, professional, and evidence-based.
                        - Do not recommend a trade unless indicators show clear confirmation
                        """
                    
            else:
                prompt = """
                You are a Senior Technical Analyst at a top hedge fund. Based on the current stock chart and indicators provided, give a **Buy, Hold, or Sell** recommendation.
                
                Instructions:
                    1. Analyze the chart using:
                        - SMA, EMA, RSI, MACD, Bollinger Bands, Volume, OBV, VWAP
                        - Support & Resistance levels, Implied Volatility
                        - ADX, ATR, Stochastic Oscillator
                    2. First, clearly state the recommendation (**Buy, Hold, or Sell**)
                    3. Then provide the rationale:
                        - Which indicators support your recommendation
                        - Any signs of trend continuation or reversal
                        - Risk factors based on volatility and volume
                """
                # prompt = f"""
                # Prompt:
                #     You are a professional Options Trader and Technical Analyst. Analyze the provided stock chart and technical indicators. 
                
                # {sr_text}

                # Instructions:
                #     1. Base you recommendation strictly on the technical data: candlestick chart, SMA, EMA, RSI, MACD, Bollinger Bands, Implied Volatility (IV), and Volume
                #     2. Start by stating:
                #         - Recommended options strategy (e.g., Iron Condor, Call Spread, Put Spread, Covered Call, Protective Put)
                #         - Whether to enter the trade (Yes or No), based only on technical evidence.
                #     3. Then , explain:
                #         - Why the selected strategy fits the current technical setup.
                #         - Suggested strick prices and expiration date, using support/resistance and current price action.
                #         - Risk/reward profile: potential gain vs. max loss, ideal entry/exit points.
                #         - Technical indicators that support the strategy (e.g., RSI > 70 = overbought, MACD crossover, IV rising, Volume)
                # Guidelines:
                #     - Be concise, professional, and evidence-based.
                #     - Do not recommend a trade unless indicators show clear confirmation
                # """
            
            # Run AI Analysis
            analysis, chart_path = ai_analysis.run_ai_analysis(fig, prompt)

            # Display Ai Analysis Results 
            st.write("**AI Analysis Results:**")
            st.write(analysis)

            # PDF Report 
            pdf = pdf_generator.PDF()
            pdf.add_page()
            pdf.add_chart(chart_path)
            pdf.add_analysis_text(analysis)

            # Save to a temporary file 
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                pdf.output(tmp_pdf.name)
                pdf_file_path = tmp_pdf.name

            # Show the PDF preview in iframe 
            with open(pdf_file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                st.markdown("### 📄 Preview AI Analysis Report")
                st.markdown(pdf_display, unsafe_allow_html=True)

            with open(pdf_file_path, "rb") as f:
                st.download_button(
                    label="⬇️Download PDF",
                    data=f.read(),
                    file_name=f'{ticker}_analysis.pdf',
                    mime="application/pdf"
                )

            # Clean up temporary file 
            os.remove(chart_path)
            os.remove(pdf_file_path)


print(f"All indicators shown are based on {interval} candles (e.g., 5-min RSI, 1-day MACD).")
