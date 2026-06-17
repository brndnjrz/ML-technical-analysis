
# AI-Powered Technical Analysis Dashboard

> A modular Streamlit application for stock and options analysis combining multi-agent AI, ensemble machine learning, and professional PDF reporting.

---

## Table of Contents

- [What It Is](#what-it-is)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Libraries Used](#libraries-used)
  - [Model Overview (AI Prediction Models)](#model-overview-ai-prediction-models)
  - [Setup Environment Using Anaconda](#setup-environment-using-anaconda)
  - [How to Run the Dashboard](#how-to-run-the-dashboard)
- [How to Use](#how-to-use)
- [Analysis Types](#analysis-types)
- [Architecture & Workflow](#architecture--workflow)
- [Project Structure](#project-structure)
- [Extensibility & Customization](#extensibility--customization)
- [Schema & Validation](#schema--validation)
- [AI Accuracy Tracking](#ai-accuracy-tracking)
- [Testing & Reliability](#testing--reliability)
- [Disclaimer & License](#disclaimer--license)
- [Appendix: Helpful Commands](#appendix-helpful-commands)

---

## What It Is

- Interactive technical charting with Plotly (candlestick, subplot overlays, support/resistance)
- Multi-agent AI analysis (Analyst, Strategy, Execution) orchestrated by `HedgeFundAI`
- Market regime detection (trend, range, event) with regime-aware decision thresholds
- Ensemble ML price prediction (Random Forest, XGBoost, CatBoost) with adaptive timeframe configuration
- Options strategy generation restricted to 6 approved, compliant strategies selected by the AI
- Strategy arbitration layer that scores and ranks candidate strategies before producing a final recommendation
- Institutional-style reports exportable to PDF
- AI accuracy tracking logged to `metrics/accuracy_log.jsonl` with a 30-day accuracy report in the sidebar

---

## Key Features

- **Three Analysis Modes:** Options Trading Strategy, Stock Buy/Hold/Sell, and Advanced Analysis (AI Ensemble) — each generates a tailored prompt and workflow.
- **Market Regime Detection:** `HedgeFundAI` classifies the current market as `trend`, `range`, or `event` before applying regime-specific decision thresholds to produce BUY / SELL / HOLD actions.
- **Multi-Agent AI System:** `AnalystAgent`, `StrategyAgent`, and `ExecutionAgent` each run independently, then `HedgeFundAI` builds consensus across all three views. Conflicts are resolved by a defined agent hierarchy (strategist for strategy, analyst for direction).
- **Strategy Arbiter:** `strategy_arbiter.py` scores candidate strategies on timeframe fit (40%), regime alignment (25%), IV fit (20%), and signal consistency (15%), then selects the best match.
- **Adaptive ML Pipeline:** Feature engineering, model hyperparameters, and cross-validation strategy all adapt to the selected interval (1m through 1d) via `adaptive_features.py`, `adaptive_models.py`, and `enhanced_validation.py`.
- **Probability Fusion:** Quantitative signals and optional Ollama vision analysis are fused using regime-aware weights before the final action is determined.
- **AI Accuracy Metrics:** Every recommendation is logged with a prediction ID. The sidebar displays a 30-day report covering directional accuracy (7-day hit rate), Brier score calibration, and per-regime breakdown.
- **Professional Reporting:** Institutional-style markdown and PDF output with schema validation and automatic fallback if the LLM output is malformed.
- **Streamlined UX:** Sidebar configuration, two-tab interface (Technical Analysis / AI Recommendation), progress bars, and contextual help throughout.
- **Robust Data Pipeline:** Market and options chain ingestion via yfinance, technical indicator calculation via pandas-ta, support/resistance detection, and options context building for AI prompts.

---

## Quick Start

1. **Create a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Or install the core packages directly:

   ```bash
   pip install streamlit plotly ollama pandas pandas-ta fpdf2 Pillow yfinance scikit-learn xgboost catboost kaleido
   ```

3. **(Optional) Run the Ollama vision model for chart-based AI analysis:**

   ```bash
   ollama run llama3.2-vision
   ```

   Keep this terminal open while using the dashboard. Vision analysis is optional; all other features work without it.

4. **Start the dashboard:**

   ```bash
   streamlit run app.py
   ```

---

### Libraries Used

| Library | Purpose |
|---------|---------|
| `streamlit` | Interactive dashboard UI |
| `yfinance` | Stock price and fundamental data |
| `pandas` | Data manipulation |
| `pandas-ta` | Technical indicator calculation |
| `plotly` | Candlestick and overlay charting |
| `ollama` | Local LLM and vision model integration |
| `scikit-learn` | Random Forest and preprocessing pipeline |
| `xgboost` | Gradient-boosted price prediction |
| `catboost` | Categorical-aware gradient boosting |
| `fpdf2`, `Pillow` | PDF report generation |
| `kaleido` | Static chart image export for AI analysis |

---

### Model Overview (AI Prediction Models)

The prediction pipeline in `src/analysis/prediction.py` uses an ensemble of three models. Weights are calculated adaptively based on recent validation performance.

- **Random Forest:** Fast, robust baseline for tabular financial data. Good for reducing noise on high-frequency intervals.
- **XGBoost:** Gradient boosting that captures complex feature interactions. Typically the strongest single model for daily data.
- **CatBoost:** Handles mixed feature types natively; strong on financial time series with categorical regime labels.
- **Ensemble:** A weighted combination of the three models. Adaptive weights (`adaptive_models.py`) and adaptive cross-validation strategy (`enhanced_validation.py`) are selected based on the active interval (e.g., `1m` uses more estimators and shallower trees than `1d`).

---

### Setup Environment Using Anaconda

1. **Download and install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install/mac-cli-install#how-do-i-verify-my-installers-integrity)**

2. **Create an environment:**

   ```bash
   conda create --name <ENV_NAME> python=3.11
   ```

3. **Activate the environment:**

   ```bash
   conda activate <ENV_NAME>
   ```

4. **Navigate to the project and install dependencies:**

   ```bash
   cd <PATH_TO_YOUR_PROJECT>
   pip install -r requirements.txt
   ```

5. **Deactivate when finished:**

   ```bash
   conda deactivate
   ```

---

### How to Run the Dashboard

1. **(Optional) Start Ollama for vision analysis:**

   ```bash
   ollama run llama3.2-vision
   ```

2. **Install required libraries** (if not already done):

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

4. **Use the dashboard:**
   - Enter a stock ticker (e.g., `AAPL`) in the sidebar
   - Select a date range, timeframe/interval, and analysis type
   - Click **"Fetch & Analyze Data"** to load price data and calculate indicators
   - Navigate to the **AI Recommendation** tab and click **"Run Analysis"** for AI-powered recommendations
   - Generate and download **PDF reports** as needed

---

## How to Use

- Enter a ticker (e.g., `AAPL`) in the sidebar
- Select a date range, timeframe/interval, and analysis type
- Click **Fetch & Analyze Data** to load data and charts
- Switch to the **AI Recommendation** tab and click **Run Analysis** for AI-powered recommendations
- Download PDF reports as needed

**Notes:**
- Strategy selection is AI-driven using the strategy arbiter and is always constrained to the 6 approved strategies
- Indicators are grouped by category (Trend, Momentum, Volatility, Volume) and pre-selected by strategy type
- Debug log level can be adjusted in the sidebar **Debug Settings** expander without restarting the app

---

## Analysis Types

The sidebar **Analysis Type** dropdown offers three modes, each generating a different prompt and workflow:

| Mode | Description |
|------|-------------|
| **Options Trading Strategy** | AI selects the optimal options strategy (Covered Calls, Cash-Secured Puts, Iron Condors, Credit Spreads, Swing Trading, Day Trading Calls/Puts) based on IV rank, regime, and signal consistency. Options priority is enabled automatically. |
| **Stock Buy/Hold/Sell** | Focuses on directional stock analysis producing a BUY / SELL / HOLD recommendation with entry price, stop loss, and take-profit targets. |
| **Advanced Analysis (AI Ensemble)** | Uses the full AI ensemble pipeline. Options priority can be toggled manually in the sidebar. |

---

## Architecture & Workflow

### Entry Point

`app.py` orchestrates the entire application: sidebar configuration, session state management, data fetching, chart rendering, and triggering the analysis workflow.

### Analysis Workflow (7 Steps)

When you click **Run Analysis**, `AnalysisWorkflowManager` in `src/analysis/workflow_manager.py` executes the following steps in order:

1. **Price Prediction** — `predict_next_period_close()` runs the ensemble ML model and returns a predicted price and confidence. The result is appended to the AI prompt.
2. **Chart Preparation** — The technical analysis chart is exported to a temporary image file via kaleido for optional vision analysis.
3. **Prompt Construction** — The prediction context is merged with the market context prompt generated by `prompt_generator.py`.
4. **AI Analysis** — `run_ai_analysis()` passes the prompt and chart to `HedgeFundAI`, which runs the three specialist agents, builds consensus, performs market regime detection, fuses quantitative and vision probabilities, and produces a final recommendation.
5. **Strategy Arbitration** — `choose_final_strategy()` in `strategy_arbiter.py` scores the AI recommendation alongside any rule-based candidate strategies and selects the best fit.
6. **Schema Validation** — The final strategy object is validated against the central JSON schema in `src/utils/ai_output_schema.py`. Malformed outputs are automatically adapted or a fallback report is generated.
7. **Completion** — Results are stored in Streamlit session state and rendered in the UI. The prediction is logged to `metrics/accuracy_log.jsonl` for later accuracy measurement.

### Session State

`AppStateManager` in `src/utils/state_manager.py` wraps all Streamlit session state reads and writes using namespaced keys defined in `SessionKeys`. This prevents key collisions and makes state transitions explicit.

---

## Project Structure

```
├── app.py                              # Main entry point and UI orchestration
├── requirements.txt
├── metrics/
│   └── accuracy_log.jsonl              # AI prediction accuracy log (auto-generated)
├── src/
│   ├── trading_strategies.py           # 6 approved strategies with timeframe configs
│   ├── plotter.py                      # Enhanced chart utilities (subplot, daily, timeframe)
│   ├── pdf_generator.py                # PDF report formatting
│   ├── pdf_utils.py                    # PDF generation and display helpers
│   ├── ai_agents/
│   │   ├── base.py                     # Shared BaseAgent with strategies_db
│   │   ├── analyst.py                  # AnalystAgent: technical indicator analysis
│   │   ├── strategy.py                 # StrategyAgent: strategy selection
│   │   ├── execution.py                # ExecutionAgent: entry/exit timing and risk
│   │   ├── hedge_fund.py               # HedgeFundAI: orchestrator and consensus engine
│   │   └── strategy_arbiter.py         # Scores and ranks candidate strategies
│   ├── analysis/
│   │   ├── indicators.py               # Technical indicator generation
│   │   ├── prediction.py               # Ensemble ML price prediction
│   │   ├── ai_analysis.py              # AI prompt execution and response formatting
│   │   ├── workflow_manager.py         # 7-step analysis workflow manager
│   │   ├── market_regime.py            # Market regime and volatility regime detection
│   │   ├── options_analysis.py         # Options chain analysis
│   │   ├── adaptive_features.py        # Timeframe-adaptive feature engineering
│   │   ├── adaptive_models.py          # Timeframe-adaptive model hyperparameters
│   │   ├── enhanced_validation.py      # Adaptive cross-validation and model selection
│   │   ├── preprocessing.py            # Preprocessing pipeline and target variable creation
│   │   └── vision_schema.py            # Vision analysis output parsing
│   ├── core/
│   │   ├── data_loader.py              # Raw market and options data ingestion
│   │   └── data_pipeline.py            # Orchestrates fetch, indicator calc, and level detection
│   ├── ui_components/
│   │   ├── sidebar_config.py           # Sidebar inputs (ticker, dates, interval, analysis type)
│   │   ├── sidebar_indicators.py       # Indicator group selection UI
│   │   ├── sidebar_stats.py            # Quick stats sidebar panel
│   │   ├── analysis_results_display.py # Analysis results rendering helpers
│   │   ├── options_analysis_display.py # Options analysis section rendering
│   │   ├── options_analyzer.py         # Options analysis logic
│   │   ├── options_strategy_selector.py
│   │   └── tabs/
│   │       ├── technical_analysis.py   # Tab 1: charts and indicators
│   │       ├── ai_recommendation.py    # Tab 2: AI recommendation trigger and display
│   │       └── analysis_results.py     # Shared analysis results component
│   └── utils/
│       ├── app_config.py               # SessionKeys, UIConfig, ProgressSteps, thresholds
│       ├── config.py                   # DEFAULT_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE
│       ├── state_manager.py            # AppStateManager: Streamlit session state wrapper
│       ├── ai_output_schema.py         # Central JSON schema for AI output validation
│       ├── metrics.py                  # AccuracyMetrics: prediction logging and reporting
│       ├── prompt_generator.py         # Analysis prompt and market context builders
│       ├── formatters.py               # Trade parameter and output formatters
│       ├── workflow_logger.py          # Structured workflow logging helpers
│       ├── logging_config.py           # Logging setup and level control
│       ├── temp_manager.py             # Temporary file lifecycle management
│       ├── vision_plotter.py           # Vision-optimized chart export utilities
│       ├── options_optimizer.py        # Options strategy optimization helpers
│       └── options_strategy_cheatsheet.py  # Options strategy reference data
└── tests/                              # Unit and integration tests
```

---

## Extensibility & Customization

- **Add a strategy:** Extend `src/trading_strategies.py` with the new strategy dict (including `Strategy`, `Description`, `Timeframe`, `Pros`, `Cons`, `When to Use`, `Timeframes` keys). Update `strategy_arbiter.py` scoring if needed and add tests.
- **Add an indicator:** Implement in `src/analysis/indicators.py`, update `sidebar_indicators.py` selection rules, and add the column name to the relevant `INDICATOR_GROUPS` in `app_config.py`.
- **Switch or tune ML models:** `src/analysis/prediction.py` uses the ensemble defined in `adaptive_models.py`. Add a new model by implementing a `get_adaptive_model_config` branch and updating the ensemble weight calculation in `calculate_adaptive_ensemble_weights`.
- **Config:** `src/utils/config.py` controls default ticker and date range. `src/utils/app_config.py` controls UI labels, thresholds (RSI, ADX, ATR, IV rank), progress step percentages, and timeframe mappings.
- **Logging:** Call `set_log_level()` from the sidebar at runtime, or configure the default in `setup_logging()` in `src/utils/logging_config.py`.

---

## Schema & Validation

- Central JSON schema for AI outputs: `src/utils/ai_output_schema.py`
- Strict required fields, optional fields allowed as `null`
- Automatic adaptation layer for flat or nested LLM outputs
- Error recovery: a readable fallback report is generated if validation fails, preventing crashes from unexpected model outputs
- Every prediction that passes validation is logged with a unique prediction ID for accuracy tracking

---

## AI Accuracy Tracking

Every time `HedgeFundAI.analyze_and_recommend()` completes, the prediction is logged to `metrics/accuracy_log.jsonl` with:

- Ticker, timestamp, and prediction ID
- Action (BUY / SELL / HOLD), confidence, and up-probability
- Market regime at time of prediction
- Key market data (price, RSI, ATR, IV rank, ADX)
- Whether vision analysis was enabled and the prompt version used

The sidebar **AI Accuracy Report** expander displays a 30-day summary including:

- Total predictions made
- 7-day directional hit rate
- Brier score (calibration quality, lower is better)
- Per-regime accuracy breakdown (trend / range / event)

---

## Testing & Reliability

- **Unit tests:** Indicators, strategy logic, schema validation (`tests/`)
- **Integration tests:** Data pipeline, AI output schema
- **CI:** Run `pytest` and linters on pull requests; smoke tests cover data fetch and report generation
- **Observability:** Centralized structured logging via `workflow_logger.py` with section timers, step logging, and prediction metrics
- **Session State:** Namespaced keys via `AppStateManager` for reproducibility and conflict-free reruns
- **Temp file cleanup:** `cleanup_old_temp_files()` runs on startup to remove stale chart exports

---

## Disclaimer & License

- **For educational purposes only** — not financial advice
- AI and LLM outputs are experimental; always verify recommendations before trading
- Every exported report includes a risk and disclaimer block
- Declare a license (e.g., MIT) in a `LICENSE` file if distributing publicly

---

## Appendix: Helpful Commands

```bash
# Run the dashboard locally
streamlit run app.py

# Run Ollama vision model (optional, keep terminal open)
ollama run llama3.2-vision

# Run tests
pytest tests/

# Install dependencies
pip install -r requirements.txt
```

---
