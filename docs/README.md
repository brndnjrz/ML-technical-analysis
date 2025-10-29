
# AI-Powered Technical Analysis Dashboard

**Executive Summary:**
This dashboard is a modular, production-grade Streamlit app for stock and options analysis. It combines multi-agent AI, robust data engineering, and professional reporting to deliver actionable, compliance-aware insights for traders, analysts, and educators.

---

## Table of contents

- [AI-Powered Technical Analysis Dashboard](#ai-powered-technical-analysis-dashboard)
  - [Table of contents](#table-of-contents)
- [What it is](#what-it-is)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
    - [Libraries Used](#libraries-used)
    - [Model Overview (AI Prediction Models)](#model-overview-ai-prediction-models)
  - [Setup Environment Using Anaconda](#setup-environment-using-anaconda)
    - [Download and install Anaconda](#download-and-install-anaconda)
  - [How to run Dashboard](#how-to-run-dashboard)
- [How to Use](#how-to-use)
- [Architecture \& Workflow (Summary)](#architecture--workflow-summary)
- [Project Structure (High Level)](#project-structure-high-level)
- [Extensibility \& Customization](#extensibility--customization)
- [Schema \& Validation](#schema--validation)
- [Testing \& Reliability](#testing--reliability)
- [Disclaimer \& License](#disclaimer--license)
  - [Appendix: Helpful Commands](#appendix-helpful-commands)

---

✅ **Change Summary:**

* Added new entries:

  * `Libraries Used`
  * `Setup Environment Using Anaconda (optional)`
  * `Download and install Anaconda`
  * `How to Use the Dashboard`
* Preserved all indentation and link anchors so clicking each heading in GitHub or VSCode will jump directly to that section.

Would you like me to also add a **“Model Overview (AI Prediction Models)”** section right after “Libraries Used” to document the purpose and differences between Random Forest, XGBoost, and CatBoost? It fits nicely there.

---


# What it is

* Interactive technical charting (Plotly)
* Multi-agent AI analysis (Analyst, Strategy, Execution, Backtest) orchestrated by HedgeFundAI
* Options strategy generation (restricted to 6 approved, compliant strategies)
* Institutional-style reports exportable to PDF
* Robust error handling and schema validation for reliability

---


# Key Features

- **Strategy Compliance:** Only 6 approved strategies (Covered Calls, Cash-Secured Puts, Iron Condors, Credit Spreads, Swing Trading, Day Trading Calls/Puts) with timeframe-specific configuration.
- **Multi-Agent AI System:** Analyst, Strategy, Execution, Backtest agents coordinated by HedgeFundAI.
- **Professional Reporting:** Institutional-style markdown and PDF output, schema validation, and fallback logic.
- **Streamlined UX:** Sidebar configuration, tabbed interface, progress/status reporting, tooltips, and contextual help.
- **Robust Data Pipeline:** Market and options ingestion, technical indicators, support/resistance detection, error resilience.
- **Extensible & Modular:** Easily add new indicators, strategies, or AI models.

---

---

# Quick Start

1. **Create virtual environment (recommended):**

  ```bash
  python -m venv .venv
  source .venv/bin/activate   # mac/linux
  .venv\Scripts\activate      # windows
  ```
2. **Install dependencies:**

  ```bash
  pip install -r requirements.txt
  # or
  pip3 install streamlit plotly ollama pandas pandas_ta fpdf kaleido yfinance scikit-learn xgboost catboost
  ```
3. **(Optional) Run Ollama vision model:**

  ```bash
  ollama run llama3.2-vision
  ```
  * Keep the Ollama terminal running while using the dashboard if you enable vision analysis.
4. **Start the dashboard:**

  ```bash
  streamlit run app.py
  ```
---


---

### Libraries Used

- `streamlit`: Interactive dashboard
- `yfinance`: Stock/fundamental data
- `pandas`: Data manipulation
- `plotly`: Charting
- `ollama`: Vision AI model integration
- `scikit-learn`, `xgboost`, `catboost`: ML models for price prediction
- `tempfile`, `base64`, `os`: File/PDF/chart handling

---

### Model Overview (AI Prediction Models)

- **Random Forest:** Fast, robust for tabular data, good baseline for price prediction.
- **XGBoost:** Gradient boosting, handles complex relationships, often best for accuracy.
- **CatBoost:** Handles categorical features natively, strong for financial time series.
- **Ensemble:** The app combines these for improved prediction reliability and confidence scoring.

---

## Setup Environment Using Anaconda

### Download and install [Anaconda](https://www.anaconda.com/download)

1. **Create an environment**

   ```bash
   conda create --name <ENV_NAME>
   ```

2. **Activate the environment**

   ```bash
   conda activate <ENV_NAME>
   ```

3. **Navigate to your project**

   ```bash
   cd <PATH_TO_YOUR_PROJECT>
   ```

   * Then follow the steps in [How to Use the Dashboard](#how-to-use-the-dashboard)

4. **Deactivate the environment**

   ```bash
   conda deactivate <ENV_NAME>
   ```

---

## How to run Dashboard

1. **Download and install [Ollama](https://ollama.com/)**

   * In a terminal, run:

     ```bash
     ollama run llama3.2-vision
     ```
   * Keep this terminal open while using the dashboard.

2. **Install required libraries**

   ```bash
   pip3 install streamlit plotly ollama pandas pandas_ta fpdf kaleido yfinance scikit-learn xgboost catboost
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

4. **Use the dashboard**

   * Enter a stock ticker (e.g., `AAPL`)
   * Select date range, timeframe, and analysis type
   * Click **“🔄 Fetch & Analyze Data”** to load data and charts
   * Click **“Run Analysis 💸”** for AI-powered recommendations
   * Generate and download **PDF reports** as needed

---


# How to Use

- Enter a ticker (e.g., `AAPL`) in the sidebar
- Select date range, timeframe, and analysis type
- Click **Fetch & Analyze Data** to load data and charts
- Click **Run Analysis** for AI-powered recommendations
- Download PDF reports as needed

**Notes:**
- Strategy selection is AI-driven and always compliant
- Indicators are grouped and pre-selected by strategy type

---


# Architecture & Workflow (Summary)

- **Main Entry:** `app.py` orchestrates UI, session state, and workflow
- **Modular Design:** `src/` contains submodules for analysis, agents, plotting, UI, and utilities
- **Session State:** Streamlit session state persists user selections, data, and results
- **Workflow:**
  1. User configures analysis in the sidebar
  2. Clicks "Fetch & Analyze Data" to load and process data
  3. Technical analysis and charts are shown in Tab 1
  4. AI analysis and trade recommendations in Tab 2
  5. Options analysis and strategy optimization as needed
  6. Export professional reports or view quick stats in the sidebar

---


# Project Structure (High Level)

```
├── app.py
├── requirements.txt
├── src/
│   ├── trading_strategies.py      # Approved strategies + timeframe configs
│   ├── data_loader.py             # Market & options ingestion
│   ├── indicators.py              # Indicator generation
│   ├── plotter.py                 # Chart utilities
│   ├── ai_analysis.py             # AI prompt/formatting
│   ├── ai_agents/                 # Analyst, strategy, execution, backtest, hedge_fund
│   ├── pdf_generator.py           # Report format + PDF export
│   ├── pdf_utils.py
│   ├── data_pipeline.py
│   ├── prediction.py              # ML models & feature engineering
│   ├── ui_components/             # Sidebar, indicators
│   ├── config.py
│   ├── logging_config.py
│   └── temp_manager.py
└── tests/                         # Unit & integration tests
```

---


# Extensibility & Customization

- **Add a strategy:** Extend `src/trading_strategies.py`, update config and tests
- **Add an indicator:** Implement in `src/indicators.py`, update sidebar selection rules
- **Switch models:** `prediction.py` supports RandomForest, XGBoost, CatBoost ensemble
- **Config:** Use `src/config.py` for timeouts, cache, endpoints, and strategy control

---


# Schema & Validation

- Central JSON schema for AI outputs: `src/utils/ai_output_schema.py`
- Strict required fields, optional fields allowed as `null`
- Automatic adaptation layer for flat/nested outputs
- Error recovery: readable fallback report if validation fails
- **Benefit:** Prevents crashes from unexpected LLM outputs, simplifies auditing

---


# Testing & Reliability

- **Unit tests:** Indicators, strategy logic, schema validation (`tests/`)
- **Integration tests:** Data pipeline, AI output schema
- **CI:** Run `pytest` + linters on PR, smoke tests for data fetch/report generation
- **Observability:** Centralized logging, metrics for analysis duration/model latency/API failures
- **Session State:** Namespaced keys for reproducibility

---


# Disclaimer & License

- **For educational purposes only** — not financial advice
- AI/LLM outputs are experimental; always verify before trading
- Every exported report includes a risk/disclaimer block
- Choose and declare a license (e.g., MIT) in `LICENSE` if open sourcing

---


## Appendix: Helpful Commands

```bash
# Run streamlit locally
streamlit run app.py

# Run Ollama (if using vision)
ollama run llama3.2-vision
```

---
