
import streamlit as st
import pandas as pd
import os
import logging
import re
from src.analysis.indicators import determine_trend
from src import plotter
from src.core.data_pipeline import fetch_and_process_data
from src.analysis.prediction import get_fundamental_metrics, predict_next_period_close
from src.analysis.ai_analysis import run_ai_analysis
from src.ai_agents.strategy_arbiter import choose_final_strategy
from src.utils.ai_output_schema import validate_ai_model_output
from src.utils.config import DEFAULT_TICKER, DEFAULT_START_DATE, DEFAULT_END_DATE
from src.utils.logging_config import setup_logging, set_log_level
from src.utils.temp_manager import temp_manager, cleanup_old_temp_files
from src.utils.metrics import get_accuracy_report
from src.utils.vision_plotter import create_vision_optimized_chart, export_chart_for_vision
from src.utils.workflow_logger import (
    log_section_start, log_section_end, 
    log_subsection_start, log_subsection_end,
    log_step, log_data_info, log_prediction, 
    log_model_performance, log_timer_start, log_timer_end,
    log_error
)
from src.pdf_utils import generate_and_display_pdf
from src.ui_components import render_sidebar_quick_stats, sidebar_config, sidebar_indicator_selection
from src.ui_components.analysis_results_display import render_analysis_results
from src.trading_strategies import strategies_data, get_strategy_by_name
from src.utils.formatters import format_analysis_text, format_professional_report
from src.utils.app_config import SessionKeys, UIConfig, ProgressSteps, TechnicalThresholds, get_user_timeframe
from src.utils.state_manager import AppStateManager
from src.ui_components.tabs.technical_analysis import render_technical_analysis_tab
from src.ui_components.tabs.ai_recommendation import render_ai_recommendation_tab
from src.ui_components.tabs.analysis_results import render_analysis_results
from src.ui_components.options_analysis_display import render_options_analysis_section
from src.analysis.workflow_manager import AnalysisWorkflowManager
from src.utils.prompt_generator import generate_analysis_prompt, build_market_context

# Setup cleaner logging for Streamlit
setup_logging(level=logging.INFO, enable_file_logging=False)

# Suppress verbose libraries
logging.getLogger('kaleido').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Clean up old temp files on startup
cleanup_old_temp_files()

# Text formatting functions moved to src.utils.formatters

# Set Up Streamlit App UI 
st.set_page_config(page_title=UIConfig.PAGE_TITLE, layout="wide")
st.title(UIConfig.APP_TITLE)
st.sidebar.header(UIConfig.SIDEBAR_HEADER)

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

# Vision Analysis Settings removed - now handled in workflow manager

# --- Accuracy Reporting Section ---
with st.sidebar.expander("📊 AI Accuracy Report"):
    if st.button("📈 View Accuracy Metrics"):
        try:
            accuracy_report = get_accuracy_report(days_back=30)
            
            if 'error' not in accuracy_report:
                st.success(f"📊 **AI Accuracy Report (Last 30 Days)**")
                
                # Overall metrics
                total = accuracy_report.get('total_predictions', 0)
                if total > 0:
                    st.metric("Total Predictions", total)
                    
                    # Directional accuracy
                    dir_acc = accuracy_report.get('directional_accuracy', {})
                    if dir_acc.get('7d_hit_rate'):
                        st.metric("7-Day Hit Rate", f"{dir_acc['7d_hit_rate']*100:.1f}%")
                    
                    # Calibration
                    calib = accuracy_report.get('calibration', {})
                    if calib.get('brier_score'):
                        brier_score = calib['brier_score']
                        st.metric("Brier Score", f"{brier_score:.3f}", 
                                 help="Lower is better (0-1 scale)")
                    
                    # By regime breakdown
                    by_regime = accuracy_report.get('by_regime', {})
                    if by_regime:
                        st.write("**By Market Regime:**")
                        for regime, data in by_regime.items():
                            regime_hit_rate = data.get('directional_accuracy', {}).get('7d_hit_rate', 0)
                            st.write(f"• {regime.title()}: {regime_hit_rate*100:.1f}% ({data['count']} predictions)")
                else:
                    st.info("No predictions available for accuracy analysis")
            else:
                st.warning("⚠️ Accuracy report unavailable - insufficient data")
                
        except Exception as e:
            st.error(f"❌ Error generating accuracy report: {e}")# Initialize state manager
state_manager = AppStateManager()

# --- Modular Sidebar ---
# Stock ticker, date range, timeframee/interval, analysis type, strategy type, technical indicators
ticker, start_date, end_date, interval, analysis_type, strategy_type, options_strategy, options_priority = sidebar_config()

# Store analysis type in session state for other components to access
state_manager.set_analysis_type(analysis_type)

# Get active indicators with unique keys
active_indicators = sidebar_indicator_selection(strategy_type, interval)


# Fetch Data Button with Enhanced Logic
# This button fetches data based on user inputs and updates the session state
if st.sidebar.button("🔄 Fetch & Analyze Data", type="primary"):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        log_section_start("DATA FETCHING WORKFLOW")
        
        status_text.text("📈 Fetching market data...")
        log_step(f"Fetching data for {ticker} from {start_date} to {end_date} with {interval} interval", "📊")
        
        start_time = log_timer_start(f"Data fetch for {ticker}")
        progress_bar.progress(25)
        
        data, levels, options_data = fetch_and_process_data(
            ticker, start_date, end_date, interval, strategy_type, analysis_type, 
            active_indicators
        )
        log_timer_end(f"Data fetch for {ticker}", start_time)
        
        progress_bar.progress(75)
        
        if data is not None:
            log_data_info(f"{ticker} stock data loaded", data)
            log_step(f"Found {len(levels.get('support', []))} support levels and {len(levels.get('resistance', []))} resistance levels", "📏")
            
            status_text.text("🔧 Calculating indicators...")
            state_manager.set_stock_data(data)
            state_manager.set_levels(levels)
            
            log_section_end("DATA FETCHING WORKFLOW")
            state_manager.set_options_data(options_data)
            state_manager.set_active_indicators(active_indicators)
            
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
if state_manager.has_stock_data():
    data = state_manager.get_stock_data()
    levels = state_manager.get_levels()
    options_data = state_manager.get_options_data()
    ticker_str = ticker.upper()

    # --- Options Strategy Analysis ---
    from src.ui_components.options_analysis_display import get_options_context_for_ai
    
    options_strategy_context = ""
    options_ai_vars = {}
    candidate_strategies = []
    features = {}
    user_timeframe = get_user_timeframe(strategy_type, interval)
    
    if analysis_type == "Options Trading Strategy":
        options_strategy_context, options_ai_vars, candidate_strategies, features = get_options_context_for_ai(
            data, options_data, ticker
        )

    # Get fundamentals with better error handling (moved before tab rendering)
    try:
        with st.spinner("Fetching fundamental data..."):
            fundamentals = get_fundamental_metrics(ticker)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch fundamental data. Using basic metrics only.")
        fundamentals = {}

    # Create tabs for different types of analysis (Options Analyzer tab removed)
    tab1, tab2 = st.tabs(["📈 Technical Analysis", "🤖 AI Recommendation"])
    
    # --- TAB 1: TECHNICAL ANALYSIS ---
    with tab1:
        render_technical_analysis_tab(
            data=data, 
            levels=levels, 
            options_data=options_data, 
            active_indicators=state_manager.get_active_indicators(),
            interval=interval,
            fundamentals=fundamentals
        )


    # Create enhanced charts for analysis
    subplot_fig, daily_fig, timeframe_fig = plotter.create_enhanced_chart(
        data=data,
        indicators=state_manager.get_active_indicators(),
        levels=levels,
        strategy_type=strategy_type,
        options_data=options_data,
        interval=interval
    )
    
    # Log indicator summary for debugging
    from src.utils.app_config import BASE_COLUMNS
    indicator_columns = [col for col in data.columns if col not in BASE_COLUMNS]
    
    logging.info(f"📊 Dashboard loaded: {len(indicator_columns)} technical indicators calculated")
    logging.debug(f"🔍 Indicator list: {', '.join(sorted(indicator_columns))}")
    
    # --- TAB 2: AI ANALYSIS ---
    with tab2:
        render_ai_recommendation_tab(state_manager)
            
    # Options analyzer functionality moved to modular components

    # Build market context and generate appropriate prompt
    from src.utils.prompt_generator import build_market_context, generate_analysis_prompt
    from src.ui_components.options_analysis_display import render_options_analysis_section
    
    market_context = build_market_context(
        ticker, interval, data, strategy_type, options_strategy, 
        state_manager.get_active_indicators(), options_strategy_context
    )
    
    prompt = generate_analysis_prompt(
        analysis_type, strategy_type, market_context, interval, ticker
    )
    
    # Display options analysis section if needed
    if analysis_type == "Options Trading Strategy":
        render_options_analysis_section(data, options_data, ticker)

    # Run AI analysis synchronously using workflow manager
    if state_manager.should_run_analysis():
        workflow_manager = AnalysisWorkflowManager(state_manager)
        
        workflow_manager.run_analysis_workflow(
            data=data,
            fundamentals=fundamentals, 
            active_indicators=state_manager.get_active_indicators(),
            ticker=ticker,
            prompt=prompt,
            options_priority=options_priority,
            candidate_strategies=candidate_strategies,
            features=features,
            user_timeframe=user_timeframe,
            daily_fig=daily_fig,
            subplot_fig=subplot_fig,
            interval=interval
        )

    if state_manager.get_ai_analysis_result() is None and state_manager.is_analysis_running():
        st.info("AI analysis started... Please wait.")
        st.spinner("AI is analyzing the market...")

    if state_manager.get_ai_analysis_result():
        analysis, chart_path, recommendation = state_manager.get_ai_analysis_result()
        
        # Use modular component to render analysis results
        render_analysis_results(
            analysis=analysis,
            chart_path=chart_path, 
            recommendation=recommendation,
            ticker=ticker,
            strategy_type=strategy_type,
            options_strategy=options_strategy,
            data=data,
            levels=levels,
            options_data=options_data,
            state_manager=state_manager
        )

        state_manager.set_analysis_running(False)


# --- SIDEBAR: QUICK STATS ---
if state_manager.has_stock_data():
    render_sidebar_quick_stats(state_manager.get_stock_data(), interval)


# Footer
st.markdown("---")
st.markdown("*This analysis is for educational purposes only. Always conduct your own research before trading.*")