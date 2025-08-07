# AI-Powered Technical Stock Analysis Dashboard


## **Overview**

This project provides an **AI-powered technical stock analysis dashboard** built with Streamlit, Plotly, and advanced technical analysis tools. The dashboard features a modern, interactive UI, dynamic strategy and indicator selection, and AI-driven market analysis with PDF reporting.

**Key Features (2025 Update):**
- Modular codebase with enhanced ML model selection (RandomForest, XGBoost, LightGBM, CatBoost)
- Real-time data fetching and analysis by default
- Combined technical and fundamental metrics for analytics and AI prediction
- Advanced feature engineering for improved prediction accuracy
- Modular UI components with streamlined configuration
- Improved error handling and logging system




### **Libraries Used**
- `streamlit`: For the interactive web dashboard
- `yfinance`: For downloading historical stock and fundamental data
- `pandas`: For data manipulation and time series
- `plotly`: For interactive candlestick and technical indicator charts
- `ollama`: For sending chart images and prompts to an AI model (LLaMA 3.2 Vision)
- `scikit-learn`: For price prediction using RandomForestRegressor
- `xgboost`: For XGBoost regression model
- `lightgbm`: For LightGBM regression model
- `catboost`: For CatBoost regression model
- `tempfile`, `base64`, `os`: For temporary file management and PDF/chart handling




## **Project Structure**

```
├── app.py                     # Main Streamlit dashboard entry point
├── src/
│   ├── data_loader.py        # Data fetching utilities
│   ├── indicators.py         # Technical indicator calculations
│   ├── plotter.py           # Charting utilities
│   ├── ai_analysis.py       # AI prompt and analysis logic
│   ├── pdf_utils.py         # PDF generation and display
│   ├── data_pipeline.py     # Data processing pipeline
│   ├── prediction.py        # ML models and feature engineering
│   ├── ui_components.py     # UI components and layouts
│   ├── config.py           # Configuration settings
│   └── trading_strategies.py # Trading strategy definitions
```

## **What the Code Does**


### **1. Streamlit App Setup & UI**
- Modern, wide-layout dashboard with streamlined controls
- Sidebar features model selection and strategy configuration
- Real-time data fetching enabled by default
- Dynamic indicator selection based on strategy type




### **2. Inputs for Stock Ticker, Date Range, and Strategy**
- Enter a stock ticker (e.g., "AAPL")
- Choose start and end dates for historical data
- Select ML model type (RandomForest, XGBoost, LightGBM, CatBoost)
- Select timeframe and analysis type
- Choose strategy type and specific options strategy if applicable




### **3. Fetch and Display Stock Data**
- Stock data is fetched using `yfinance` for the selected ticker, date range, and interval.
- Data is stored in `st.session_state` for persistence and fast UI updates.





### **4. Technical & Fundamental Metrics, Charting**
- Interactive candlestick charts are created with Plotly.
- Sidebar allows dynamic selection of technical indicators, grouped by category (momentum, trend, volatility, volume, oscillators).
- Recommended indicators are pre-selected based on chosen strategy.
- Indicators include: SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, ATR, ADX, Stochastic, OBV, and more.
- **Fundamental metrics** (EPS, P/E Ratio, Revenue Growth, Profit Margin) are fetched via yfinance and combined with technicals in the dashboard and for AI analysis.





### **5. AI-Powered Analysis & Price Prediction**
- Advanced feature engineering for ML models
- Multiple model options with optimized configurations
- Automatic handling of missing values and data cleaning
- Real-time prediction confidence scoring
- Contextual AI analysis based on prediction results
- Results are displayed in the dashboard.
- Users can generate a comprehensive PDF report (with chart, analysis, indicators, and risk metrics) for preview and download.





### **6. Temporary File & State Management**
- Temporary files for charts and PDFs are managed and cleaned up automatically.
- All user selections and results are managed via Streamlit session state for a seamless experience.
- Modular session state keys for stock data, levels, options data, and active indicators.





## **Modularization & Maintainability**

- All major logic is separated into modules in `src/` for easy maintenance and extension.
- UI components (sidebar, quick stats) are reusable and easy to update.
- Data pipeline and prediction logic are decoupled from the main app for clarity.

## **Use Cases**

1. **Stock Trading and Investment:**
   - Analyze historical and real-time stock data with advanced technical indicators and AI-driven insights.
2. **Options Strategy Planning:**
   - Select and evaluate options strategies (Iron Condor, Covered Calls, etc.) with AI recommendations.
3. **Educational Tool:**
   - Teach technical analysis, charting, and options strategies interactively.
4. **AI-Assisted Decision Making:**
   - Use LLaMA 3.2 Vision to get actionable, explainable trade recommendations.
5. **Professional Reporting:**
   - Generate and download comprehensive PDF reports for research or sharing.


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

1. Download and install [Ollama](https://ollama.com/)
   - In a terminal, run:
     ```bash
     ollama run llama3.2-vision
     ```
   - Keep this terminal open while using the dashboard.
2. In a new terminal, install the required libraries:
   ```bash
   pip install streamlit plotly ollama pandas yfinance scikit-learn xgboost lightgbm catboost
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Use the dashboard:
   - Enter a stock ticker (e.g., AAPL)
   - Select date range, timeframe, and analysis type
   - Choose or accept recommended indicators
   - Select options strategies if desired
   - Click "🔄 Fetch & Analyze Data" to load data and charts
   - Click "Run Analysis 💸" for AI-powered recommendations
   - Generate and download PDF reports as needed


## **Disclaimer**

### For Educational Purposes Only
- This dashboard does not constitute financial or investing advice.
### AI/LLM Technology is Experimental
- Outputs may contain inaccuracies or misleading information.
### Use Critical Thinking
- Always verify conclusions and make informed decisions.