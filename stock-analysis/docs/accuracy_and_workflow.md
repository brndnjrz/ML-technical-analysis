# AI Accuracy Optimization & Workflow Reference

This document is the authoritative reference for the accuracy optimization system and the end-to-end workflow of the AI trading platform. It covers every implemented subsystem, the phase-by-phase execution flow, usage examples, and configuration options.

For the visual execution flow, see [flowchart.md](flowchart.md).

---

## System Overview

The AI trading system is an enterprise-grade accuracy-optimized trading platform featuring:

- **Multi-Agent Intelligence**: 4 specialized AI agents working in consensus
- **Regime-Aware Analysis**: Automatic market condition detection and adaptation
- **Vision-Enhanced Processing**: AI chart analysis with structured validation
- **Accuracy Optimization**: Comprehensive prediction tracking and calibration
- **Professional Risk Management**: Institutional-grade risk controls and reporting

---

## Implemented Subsystems

### 1. Metrics Tracking & Measurement (`src/utils/metrics.py`)

Logs every AI prediction with contextual data and tracks performance across multiple time horizons and regimes.

**Metrics tracked:**
- Directional hit rate (1, 3, 7-day horizons)
- Range prediction accuracy
- Brier score and reliability (confidence calibration)
- Risk assessment accuracy
- Strategy P&L by strategy type
- Regime-specific accuracy breakdown

**Storage:** JSONL format at `metrics/accuracy_log.jsonl` for fast analytics. Evaluation window defaults to 30 days.

```python
from src.utils.metrics import log_prediction, get_accuracy_report

# Log a prediction
prediction_id = log_prediction(ticker, recommendation, regime, market_data)

# Get accuracy report
report = get_accuracy_report(days_back=30)
```

### 2. Market Regime Detection (`src/ai_agents/hedge_fund.py`)

Automatically detects the current market regime using ADX, moving average slopes, and volatility ratios, then categorizes the market as `trend`, `range`, or `event`.

**Regime logic:**

| Regime | Detection Criteria | Quant Weight | Vision Weight |
|--------|-------------------|--------------|---------------|
| Trend  | ADX >= 20 + significant MA slope | 70% | 30% |
| Range  | Low ADX + oscillatory price action | 45% | 55% |
| Event  | High IV/RV ratio or near earnings | 60% | 40% |

```python
# Automatic market regime detection
regime = hedge_fund.detect_regime(data, ticker, options_data)
```

### 3. Structured Vision Analysis (`src/analysis/vision_schema.py`)

Enforces a strict JSON schema for vision model output, validates price levels against reasonable bounds, and provides structured error handling and fallbacks.

**Schema validation features:**
- Pydantic validation with automatic type checking
- Price level sanity checks (±20% bounds)
- Confidence scoring with uncertainty handling
- Pattern recognition standardization

**Vision output schema:**
```json
{
    "trend": "bullish|bearish|neutral",
    "support": [145.30, 142.80, 140.50],
    "resistance": [152.40, 155.20, 158.00],
    "risk": "low|medium|high",
    "confidence": 0.75,
    "pattern": "ascending_triangle",
    "breakout_probability": 0.68
}
```

### 4. Regime-Aware Probability Fusion (`src/ai_agents/hedge_fund.py`)

Combines quantitative and vision analysis with regime-specific weights, applies calibration adjustments based on historical performance, and handles confidence scaling and uncertainty quantification.

**Fusion methodology:**
- Dynamic weighting by regime (see table in section 2 above)
- Confidence scaling: vision weight adjusted by AI confidence level
- Calibration adjustments: historical performance corrections applied
- Uncertainty propagation throughout the pipeline

```python
# Combine quantitative and vision signals with regime weighting
fused_probabilities = hedge_fund.fuse_agent_probabilities(
    analyst_view=quant_analysis,
    vision_analysis=vision_results,
    regime=detected_regime
)
```

### 5. Decision Thresholds & No-Trade Zones

Implements explicit probability thresholds for trade entry, creates no-trade zones to avoid marginal decisions, and applies regime-specific decision criteria.

| Regime | Bullish Entry | Bearish Entry | No-Trade Zone | Min Confidence |
|--------|---------------|---------------|---------------|----------------|
| Trend  | >= 0.58      | <= 0.42      | 0.47–0.53    | 0.60          |
| Range  | >= 0.62      | <= 0.38      | 0.45–0.55    | 0.50          |
| Event  | >= 0.65      | <= 0.35      | 0.40–0.60    | 0.70          |

```python
# Regime-specific decision thresholds
thresholds = {
    'trend': {'buy': 0.58, 'sell': 0.42, 'no_trade': (0.47, 0.53)},
    'range': {'buy': 0.62, 'sell': 0.38, 'no_trade': (0.45, 0.55)},
    'event': {'buy': 0.65, 'sell': 0.35, 'no_trade': (0.40, 0.60)}
}
```

### 6. Enhanced Chart Generation (`src/utils/vision_plotter.py`)

Creates deterministic, labeled charts optimized for AI vision processing with consistent theme, fonts, and layout.

**Vision optimization features:**
- Fixed 800x800 resolution, <250KB WebP compression
- Clear panel labeling (PRICE, RSI, VOLUME)
- Distinct color coding for support/resistance levels
- Context annotations: current price, RSI condition, trend direction
- Metadata watermarking with key market context

```python
# Create deterministic, AI-optimized charts
vision_chart = create_vision_optimized_chart(
    data=stock_data,
    indicators=active_indicators,
    levels=support_resistance_levels,
    strategy_type=selected_strategy
)
```

### 7. Options Strategy Optimization (`src/utils/options_optimizer.py`)

Scores option strategies across strike/expiry grids, calculates expected value using scenario analysis, and includes transaction costs and slippage.

**Grid scoring features:**
- Black-Scholes approximation for option pricing
- Multiple scenario P&L calculation
- Risk-adjusted scoring methodology
- Automatic strike/expiry optimization
- Liquidity analysis: bid-ask spread and volume considerations

### 8. Schema Validation & Adaptation (`src/utils/ai_output_schema.py`)

Enforces consistent structure for AI outputs, provides intelligent adaptation for non-conforming data, and ensures graceful degradation with sensible fallbacks.

**Validation system:**
- Comprehensive JSON schema definition with descriptive field documentation
- Type validation allowing both strict types and null values where appropriate
- Required field enforcement with flexibility for optional data
- Support for complex nested objects including strategy recommendations

**Adaptation framework:**
- Structure transformation: intelligent adaptation between flat and nested data structures
- Default values: automatic fallback to sensible defaults when fields are missing
- Type coercion: smart handling of type inconsistencies for numeric fields
- Error recovery: graceful handling of validation errors with informative messages

**Benefits:**
- Prevents crashes from malformed AI outputs
- Maintains consistency across different models and agents
- Improves reliability in production environments
- Self-documenting data contracts through schema definitions

```python
from src.utils.ai_output_schema import validate_ai_model_output

# Validate and potentially adapt AI model output
try:
    validation_result = validate_ai_model_output(strategy_data, ticker="SPY")
    if validation_result and not isinstance(validation_result, bool):
        # Use the adapted data if validation transformed the structure
        strategy_data = validation_result
except Exception as e:
    # Fallback handling with appropriate default values
    logging.error(f"Validation error: {str(e)}")
```

### 9. Integrated Accuracy Reporting (`app.py`)

Real-time accuracy metrics displayed in the Streamlit sidebar, including regime-specific performance breakdown, calibration monitoring, and historical hit rate trends.

---

## End-to-End Workflow

### Phase 1: System Initialization & Configuration

The dashboard loads with an enhanced sidebar. The accuracy metrics sidebar displays real-time performance data on startup.

**Configuration options:**
- **Ticker selection**: Stock symbol with validation
- **Time parameters**: Start/end dates, interval selection
- **Analysis type**: Stock trading vs options strategies
- **Technical indicators**: Strategy-specific indicator selection
- **AI settings**: Vision analysis enable/disable, timeout controls, options priority toggle

### Phase 2: Enhanced Data Pipeline

```python
# Enhanced data fetching with comprehensive validation
data, levels, options_data = fetch_and_process_data(
    ticker, start_date, end_date, interval,
    strategy_type, analysis_type, active_indicators
)
```

**Data components collected:**
1. Market data: OHLCV with dividend adjustments
2. Technical indicators: 30+ indicators calculated with error handling
3. Options chain: real-time IV, HV, and Greeks data
4. Support/resistance: automated level detection
5. Fundamental data: EPS, P/E, revenue growth metrics

**Quality assurance:** data validation, missing data interpolation, outlier detection, and schema validation applied to all datasets.

### Phase 3: Vision-Optimized Chart Generation

Charts are generated with deterministic themes and metadata watermarks before AI analysis begins. See section 6 above for chart configuration details.

### Phase 4: Market Regime Detection

The regime is detected automatically before any agent analysis runs. See section 2 above for detection criteria and regime-specific weights.

### Phase 5: Multi-Agent AI Analysis

```python
# Initialize hedge fund AI with 4 specialized agents
hedge_fund = HedgeFundAI(config={
    'consensus_threshold': 0.6,
    'risk_limits': {
        'max_position_size': 0.1,
        'max_portfolio_vol': 0.15,
        'min_confidence': 0.5
    }
})
```

**Agent specializations:**

1. **AnalystAgent** (`src/ai_agents/analyst.py`): technical analysis and market research, indicator interpretation, market structure analysis
2. **StrategyAgent** (`src/ai_agents/strategy.py`): strategy selection and optimization, risk-return analysis, portfolio context integration
3. **ExecutionAgent** (`src/ai_agents/execution.py`): entry/exit timing optimization, order type selection, slippage and transaction cost modeling
4. **BacktestAgent**: historical performance validation, risk metric calculation, strategy robustness testing

### Phase 6: Consensus Decision Engine

```python
# Multi-agent consensus with conflict resolution
consensus = hedge_fund.build_consensus(
    analyst_view, strategy_recommendation,
    execution_signals, backtest_results
)
```

**Consensus process:**
- Agreement threshold: 60% minimum consensus required
- Conflict resolution: automatic arbitration using risk-adjusted scores
- Quality gates: minimum confidence and risk limit validation
- Override mechanism: risk committee can override low-confidence decisions

### Phase 7: Strategy Arbiter & Schema Validation

The Strategy Arbiter scores candidate strategies, filters by timeframe, and matches against the detected market regime to select the highest-scoring strategy. The selected output then passes through schema validation (see section 8 above). If validation fails, the adaptation layer attempts to transform the data; if adaptation fails, default fallbacks are used.

### Phase 8: Vision Analysis Integration

```python
# AI vision analysis with schema validation
vision_analysis = analyze_chart_with_vision(
    chart_image, timeout=vision_timeout,
    schema_validation=True
)
```

**Vision processing pipeline:**
1. Chart analysis: AI processes the optimized technical chart
2. JSON extraction: structured data extraction with regex patterns
3. Schema validation: Pydantic models enforce data consistency
4. Fallback handling: natural language parsing if JSON extraction fails
5. Price validation: sanity checks against ±20% bounds
6. Confidence scoring: uncertainty quantification for reliability

If vision analysis is disabled, the system proceeds with quantitative-only analysis and skips the fusion step.

### Phase 9: Regime-Aware Probability Fusion

Quantitative and vision signals are combined using regime-specific weights. See sections 4 and 5 above for fusion weights and decision thresholds.

### Phase 10: Strategy Optimization

**Options priority mode:**
```python
if options_priority:
    optimal_strategy = options_optimizer.score_option_grid(
        strikes=available_strikes,
        expiries=available_expiries,
        scenarios=price_scenarios,
        market_data=current_conditions
    )
```

**Strategy selection process:**
1. Grid scoring: expected value across strike/expiry combinations
2. Risk adjustment: Sharpe ratio and maximum drawdown analysis
3. Transaction costs: realistic slippage and commission modeling
4. Liquidity analysis: bid-ask spread and volume considerations
5. Portfolio context: correlation and diversification effects

When options priority is off, the system uses stock strategy focus with buy/hold/sell signals, stop-loss optimization, and position sizing.

### Phase 11: Accuracy Tracking & Logging

```python
# Log every prediction with complete context
prediction_id = log_prediction(
    ticker=ticker,
    recommendation=final_recommendation,
    regime=market_regime,
    market_data=current_conditions,
    vision_enabled=vision_analysis_used,
    prompt_version="accuracy_optimized_v1"
)
```

See section 1 above for the full list of tracked metrics.

### Phase 12: Professional Reporting

```python
# Generate professional trade report
professional_report = format_professional_report(
    analysis=ai_analysis,
    recommendation=final_recommendation,
    ticker=ticker,
    strategy_type=strategy_type,
    market_data=current_data,
    regime=market_regime
)
```

**Report components:**
- Executive summary: action, confidence, strategy overview
- Market analysis: current conditions and regime classification
- Risk assessment: position sizing and risk metrics
- Trade parameters: entry, exit, stop-loss levels
- Options strategy: strike/expiry optimization if applicable
- Accuracy context: historical performance and calibration

Optionally generates a PDF. After display, the sidebar accuracy dashboard updates with real-time hit rates, regime performance, calibration metrics, and historical trends.

### Phase 13: Continuous Learning & Optimization

```python
# Continuous accuracy monitoring
accuracy_report = get_accuracy_report(
    days_back=30,
    by_regime=True,
    by_strategy=True
)

# Later, when outcome is known
record_outcome(prediction_id, {
    'directional_correct_7d': True,
    'inside_range_7d': False,
    'max_adverse_move_pct': -0.03,
    'realized_pnl_pct': 0.05
})
```

**Self-optimization features:**
1. Outcome recording: automatic result tracking
2. Performance analysis: pattern recognition in successes and failures
3. Calibration adjustment: threshold and weight optimization over time
4. Strategy evolution: adaptive algorithm improvement
5. Regime learning: market condition pattern recognition

---

## Expected Performance Improvements

Based on the implemented optimizations:

| Metric | Expected Improvement |
|--------|---------------------|
| Directional hit rate (7-day) | +6–10% over baseline |
| False positive trades | 20% reduction via no-trade zones |
| Range accuracy on range-bound days | >= 60% |
| Brier score (confidence calibration) | 15%+ improvement |
| Risk management | Better volatility prediction alignment |

**Success KPIs to track:**
1. 7-day directional accuracy > Baseline + 8%
2. Brier score improvement > 15%
3. False positive reduction > 20%
4. Range strategy hit rate > 60%
5. Confidence calibration > 0.8 reliability score

---

## Implementation Status

### Completed
- Comprehensive metrics tracking system (`src/utils/metrics.py`)
- Market regime detection algorithm (`src/ai_agents/hedge_fund.py`)
- Structured vision analysis with schema validation (`src/analysis/vision_schema.py`)
- Regime-aware probability fusion (`src/ai_agents/hedge_fund.py`)
- Decision thresholds with no-trade zones (`src/ai_agents/hedge_fund.py`)
- Enhanced deterministic chart generation (`src/utils/vision_plotter.py`)
- Options strategy grid optimization (`src/utils/options_optimizer.py`)
- AI output schema validation and adaptation (`src/utils/ai_output_schema.py`)
- Integrated accuracy reporting UI (`app.py`)

### Next Steps (Phase 2)
- Backtesting harness with frozen data
- A/B prompt testing framework
- Isotonic regression calibration
- Weekly error analysis automation
- Confidence band display improvements

---

## Module Architecture

```
src/
├── utils/
│   ├── metrics.py              # Accuracy tracking & measurement
│   ├── vision_plotter.py       # Deterministic chart generation
│   ├── options_optimizer.py    # Strike/expiry optimization
│   └── ai_output_schema.py     # Schema validation for AI outputs
├── analysis/
│   └── vision_schema.py        # Structured vision output validation
└── ai_agents/
    ├── analyst.py              # Market analysis agent
    ├── strategy.py             # Strategy selection agent
    ├── execution.py            # Execution timing agent
    ├── hedge_fund.py           # Orchestrator: regime detection & fusion
    └── strategy_arbiter.py     # Strategy scoring and selection
```

---

## Monitoring & Maintenance

The system automatically logs prediction failures and schema validation errors, regime detection anomalies, fusion weight adjustments, and decision threshold violations. Monitor these logs for system health and performance optimization opportunities.

**System health indicators:**
- Prediction accuracy trends: monitor hit rate changes over time
- Calibration drift: watch confidence vs. outcome alignment
- Regime detection: validate market classification accuracy
- Performance attribution: track success by component

**Optimization levers:**
- Threshold tuning: adjust decision boundaries based on performance data
- Weight optimization: refine regime-specific fusion weights
- Strategy evolution: add new strategies based on market patterns
- Vision enhancement: improve chart analysis prompts and validation schemas
