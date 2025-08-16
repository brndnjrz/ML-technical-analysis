# =========================
# Imports
# =========================
import pandas as pd
import streamlit as st
import traceback
from . import data_loader
from ..analysis import indicators

# =========================
# Main Data Pipeline Function
# =========================
def fetch_and_process_data(
    ticker,
    start_date,
    end_date,
    interval,
    strategy_type,
    analysis_type,
    active_indicators
):
    """
    Fetch and process stock data, calculate indicators, support/resistance, and options metrics.
    Returns processed data, levels, and options data (if applicable).
    """
    try:
        # Fetch raw stock data
        data = data_loader.fetch_stock_data(ticker, start_date, end_date, interval)
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            st.error(f"❌ No valid data for {ticker}. Please check ticker, date range, and market hours.")
            return None, None, None

        # Calculate technical indicators
        try:
            data_with_indicators = indicators.calculate_indicators(
                data,
                timeframe=interval,
                strategy_type=strategy_type,
                selected_indicators=active_indicators
            )
        except Exception as indicator_error:
            st.error(f"❌ Error calculating indicators: {str(indicator_error)}")
            data_with_indicators = data

        # Detect support/resistance levels
        try:
            method = "advanced" if "Long-Term" in str(strategy_type) else "quick"
            levels = indicators.detect_support_resistance(data_with_indicators, method=method)
        except Exception as level_error:
            st.error(f"❌ Error detecting levels: {str(level_error)}")
            levels = {'support': [], 'resistance': []}

        # Options metrics (if applicable)
        options_data = None
        if analysis_type == "Options Trading Strategy":
            try:
                iv_data = indicators.calculate_iv_metrics(ticker, data_with_indicators)
                options_flow = indicators.get_options_flow(ticker)  # Always fetch real-time options flow
                options_data = {"iv_data": iv_data, "options_flow": options_flow}
            except Exception as options_error:
                st.warning(f"⚠️ Options data unavailable: {str(options_error)}")
                options_data = None

        return data_with_indicators, levels, options_data

    except Exception as e:
        st.error(f"❌ Error in fetch_and_process_data: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None
