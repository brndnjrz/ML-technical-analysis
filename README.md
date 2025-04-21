# AI-Powered Technical Stock Analysis Dashboard

## **Overview**
This Python script creates an **AI-powered technical stock analysis dashboard** using Streamlit, Plotly, and various technical analysis tools. Below is a detailed explanation of what the code does and its use cases.


### **Libraries Used**
- `streamlit`: Used to build the web interface
- `yfinance`: Used to download historical stock data from Yahoo Finance
- `pandas`: Used for data manipulation and working with time series.
- `plotly.graph_objects`: Used to create the interactive candlestick chart. 
- `ollama`: Used to send images and prompts to an AI model (LLaMA 3.2 Vision) for stock chart analysis 
- `tempfile`, `base4`, `os`: Handle temporary file creation, encoding chart images, and file cleanup.



## **What the Code Does**

### **1. Streamlit App Setup**
- The app uses **Streamlit** for a web-based dashboard.
- The layout is set to wide using `st.set_page_config(layout="wide")`.
- A title is displayed: *"AI-Powered Technical Stock Analysis Dashboard"*.
- A sidebar is included for configuration options.



### **2. Inputs for Stock Ticker and Date Range**
- Users can enter a stock ticker (e.g., "AAPL") in a text box.
- Start and end dates can be selected for fetching historical stock data.
- The `st.sidebar.button("Fetch Data")` button triggers the fetching of data using **Yahoo Finance** (`yfinance`).



### **3. Fetch and Display Stock Data**
- The stock data is retrieved using `yf.download()` for the specified date range.
- The data is stored in `st.session_state` for state persistence.



### **4. Plot a Candlestick Chart**
- A candlestick chart of the stock's open, high, low, and close prices is created using **Plotly**.
- Users can select from technical indicators (20-Day SMA, EMA, Bollinger Bands, VWAP) via a sidebar multiselect widget.
- The selected indicators are overlaid on the candlestick chart using helper functions:
  - **SMA (Simple Moving Average)**: 20-day moving average of closing prices.
  - **EMA (Exponential Moving Average)**: Similar to SMA but gives more weight to recent prices.
  - **Bollinger Bands**: Bands around the SMA showing volatility.
  - **VWAP (Volume Weighted Average Price)**: Weighted average price based on volume.



### **5. AI-Powered Analysis**
- Uses the `ollama` library (LLaMA 3.2 Vision) for analyzing stock charts.
- When users click "Run AI Analysis":
  - The candlestick chart is saved as an image in a temporary file.
  - The image is encoded to Base64 and sent to the AI model along with a prompt.
  - The AI model analyzes the chart and provides a **buy/hold/sell recommendation** with reasoning.
- The result is displayed on the Streamlit dashboard.



### **6. Temporary File Management**
- Temporary files used for saving the chart are cleaned up after use to avoid clutter.



## **Use Cases**

1. **Stock Trading and Investment:**
   - Traders and investors can use the dashboard to analyze historical stock data and make informed decisions based on technical indicators.

2. **Educational Tool:**
   - Teaching beginners about candlestick charts and technical indicators like SMA, EMA, and Bollinger Bands.

3. **AI-Assisted Decision Making:**
   - Leverages LLaMA 3.2 Vision to assist traders by providing actionable recommendations (buy/hold/sell) based on chart patterns.

4. **Prototyping AI in Finance:**
   - Developers can use this as a prototype for integrating AI with financial tools.

5. **Data Visualization:**
   - Allows users to visualize complex stock data trends and patterns in an interactive manner.


## **Setup Environment Using Anaconda(optional)**
### Download and install [Anaconda](https://www.anaconda.com/download)
1. Create an environment 
```bash
conda create --name <ENV_NAME>
```    
2. To activate this environment
```bash
conda activate <ENV_NAME>
```
3. To run the app in this environment
```bash
cd <PATH_TO_YOU_PROJECT>
```
   - Then follow steps in [How to Use the Dashboard](#how-to-use-the-dashboard)

4. To deactivate an an active environment
```bash
conda deactivate <ENV_NAME>
```


## **How to Use the Dashboard**

1. Download [Ollama](https://ollama.com/) 
   - Once installed open up terminal and run the following command (keep open)
    ```bash
    ollama run llama3.2-vision
    ```
2. Open another terminal and install the required libraries:
   ```bash
   pip install streamlit plotly ollama pandas yfinance==0.2.40
   ```
3. Run the app 
    ```bash
    streamlit run app.py
    ```
4. Interact with the dashboard:
   - Input a stock ticker (e.g., AAPL)
   - Select a date range.
   - Choose indicators to overlay on the chart.
   - Run AI analysis to get recommendations.

## **Disclaimer**

### For Educational Purposes Only
- Does not constitute financial or investing advice 
### AI/LLM Technology is Experimental
- Outputs may contain inaccuracies or misleading information
### Use Critical Thinking
- Always verify conclusions and make informed decisions