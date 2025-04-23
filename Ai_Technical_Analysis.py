## NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
from pdf_generator import PDF

# Set up Streamlit App UI
st.set_page_config(layout="wide")   # Page layout set to full width
st.title("AI-Powered Technical Stock Analysis Dashboard")   # Displays the title at the top 
st.sidebar.header("Configuration")  # Header in the sidebar

# Sidebar Input 
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")  # Stock ticker input 
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))    # Allows user to select start and end date
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# Fetch and Store Stock Data
if st.sidebar.button("Fetch Data"): # When user clicks Fetch Data
    stock = yf.Ticker(ticker)   # Creates a ticker object 
    # Data is stored in st.session_state
    st.session_state["stock_data"] = stock.history(start=start_date, end=end_date, auto_adjust=False)   # Fetches historical data between selected dates
    # st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
    st.success("Stock data loaded successfully!")

# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Plot Candlestick Chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,   # data.index represents the x-axis (dates)
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"  # Replace "trace 0" with "Candlestick"
        )
    ])

    # Sidebar: Select Technical Indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(    # Uses a multi-select dropdown
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    # Function to Compute and Add Selected Indicators
    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            
            # data['Close']: access the column of closing prices 
            # .rolling(window=20).mean(): computes the 20-day simple moving average of the closing prices.
            # go.Scatter(...): creates a line plot(mode='lines') of this moving average
            # fig.add_trace(...): adds the SMA line to the candlestick chart
            # Why its useful? 
            #     Helps traders identify price trends over time 
            
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            
            # .ewm(span=20).mean(): computes the exponential moving average using a 20-day span. Unlike SMA, EMA gives more weight to recent prices
            # Why its useful? 
            #     More responsive to recent price changes than SMA
            
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            
            # Calculates the Bollinger Bands
            #     sma: 20-day simple moving average
            #     std: 20-day standard deviation of closing prices
            #     bb_upper: Upper band = SMA + 2 * std dev
            #     bb_lower: Lower band = SMA + 2 * std dev
            # Adds both upper and lower bands to the chart as separate lines
            # Why its useful?
            #     Helps visualize price volatility and potential breakout points 
            
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            
            # VWAP formula
            #     VWAP = ϵ(Close x Volume) / ϵ(Volume)
            # .cumsum(): means cumulative sum over the entire dataset
            # VWAP: gives the average price at which a stock has traded throughout the day, based on both price and volume
            # Plots VWAP as a line on the chart
            # Why its useful?
            #     Institutional traders use it to determine whether they are getting a good price
            
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

    # Add selected indicators to the chart
    for indicator in indicators:
        add_indicator(indicator)

    # Final Chart Formatting and display 
    fig.update_layout(xaxis_rangeslider_visible=False)  # Hides the x-axis range slider 
    st.plotly_chart(fig)

    # Analyze chart with LLaMA 3.2 Vision
    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):    # Adds a button that triggers the following 
        with st.spinner("Analyzing the chart, please wait..."):
            # Saves the Plotly chart as a PNG in a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name

            # Read image and encode to Base64
            with open(tmpfile_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Prepare AI analysis request
            messages = [{
                'role': 'user',
                'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                            Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                            Base your recommendation only on the candlestick chart and the displayed technical indicators.
                            First, provide the recommendation, then, provide your detailed reasoning.
                """,
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision', messages=messages)

            # Display AI analysis result
            st.write("**AI Analysis Results:**")
            st.write(response["message"]["content"])

            # PDF Report
            pdf =PDF()
            pdf.add_page()
            pdf.add_chart(tmpfile_path)
            pdf.add_analysis_text(response["message"]["content"])

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
            os.remove(tmpfile_path)
            os.remove(pdf_file_path)