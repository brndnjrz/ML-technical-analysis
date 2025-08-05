import pandas as pd
import streamlit as st
import traceback
from src import data_loader, indicators

def fetch_and_process_data(ticker, start_date, end_date, interval, strategy_type, analysis_type, fetch_realtime, active_indicators):
    """Fetch and process data with strategy-specific calculations"""
    try:
        data = data_loader.fetch_stock_data(ticker, start_date, end_date, interval)
        if data is None:
            st.error(f"❌ No data available for {ticker}. Please check:")
            st.error("- Ticker symbol is correct")
            st.error("- Date range is valid") 
            st.error("- Market was open during selected period")
            return None, None, None
        if data.empty:
            st.error(f"❌ Empty dataset returned for {ticker}")
            return None, None, None
        if not isinstance(data, pd.DataFrame):
            st.error(f"❌ Expected DataFrame, got {type(data)}")
            return None, None, None
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
        try:
            levels = indicators.detect_support_resistance(
                data_with_indicators, 
                method="advanced" if "Long-Term" in str(strategy_type) else "quick"
            )
        except Exception as level_error:
            st.error(f"❌ Error detecting levels: {str(level_error)}")
            levels = {'support': [], 'resistance': []}
        if analysis_type == "Options Trading Strategy":
            try:
                iv_data = indicators.calculate_iv_metrics(ticker, data_with_indicators)
                options_flow = indicators.get_options_flow(ticker) if fetch_realtime else None
                return data_with_indicators, levels, {"iv_data": iv_data, "options_flow": options_flow}
            except Exception as options_error:
                st.warning(f"⚠️ Options data unavailable: {str(options_error)}")
                return data_with_indicators, levels, None
        return data_with_indicators, levels, None
    except Exception as e:
        st.error(f"❌ Error in fetch_and_process_data: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None
