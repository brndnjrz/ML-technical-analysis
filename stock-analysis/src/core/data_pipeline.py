# =========================
# Imports
# =========================
import pandas as pd
import streamlit as st
import traceback
import logging
import time
from datetime import datetime
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
    logging.info(f"\n{'=' * 60}")
    logging.info(f"📊 DATA PIPELINE: Processing {ticker} from {start_date} to {end_date}")
    logging.info(f"{'=' * 60}")
    start_time = time.time()
    
    try:
        # Fetch raw stock data
        logging.info(f"⬇️ Fetching data for {ticker} with {interval} interval...")
        fetch_start = time.time()
        data = data_loader.fetch_stock_data(ticker, start_date, end_date, interval)
        fetch_time = time.time() - fetch_start
        
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            logging.error(f"❌ No valid data for {ticker}. Please check ticker, date range, and market hours.")
            st.error(f"❌ No valid data for {ticker}. Please check ticker, date range, and market hours.")
            return None, None, None
        
        logging.info(f"✅ Data fetched in {fetch_time:.2f}s - {len(data)} rows from {data.index.min().date()} to {data.index.max().date()}")
        
        # Calculate technical indicators
        try:
            logging.info(f"📈 Calculating {len(active_indicators)} technical indicators...")
            indicators_start = time.time()
            data_with_indicators = indicators.calculate_indicators(
                data,
                timeframe=interval,
                strategy_type=strategy_type,
                selected_indicators=active_indicators
            )
            indicators_time = time.time() - indicators_start
            logging.info(f"✅ Indicators calculated in {indicators_time:.2f}s")
            
            # Log added columns
            added_columns = set(data_with_indicators.columns) - set(data.columns)
            if added_columns:
                logging.info(f"📊 Added {len(added_columns)} indicator columns: {', '.join(sorted(added_columns))}")
                
            # Check for and log NaN values
            for column in data_with_indicators.columns:
                nan_count = data_with_indicators[column].isna().sum()
                if nan_count > 0:
                    logging.info(f"🔍 Found {nan_count} NaN values in column {column}")
                    
        except Exception as indicator_error:
            logging.error(f"❌ Error calculating indicators: {str(indicator_error)}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            st.error(f"❌ Error calculating indicators: {str(indicator_error)}")
            data_with_indicators = data

        # Detect support/resistance levels
        try:
            logging.info(f"🎯 Detecting support/resistance levels...")
            levels_start = time.time()
            method = "advanced" if "Long-Term" in str(strategy_type) else "quick"
            levels = indicators.detect_support_resistance(data_with_indicators, method=method)
            levels_time = time.time() - levels_start
            
            support_count = len(levels.get("support", []))
            resistance_count = len(levels.get("resistance", []))
            logging.info(f"✅ Detected {support_count} support and {resistance_count} resistance levels in {levels_time:.2f}s")
            
            # Log actual levels if available
            if support_count > 0:
                logging.info(f"📉 Support levels: {', '.join([f'${x:.2f}' for x in levels.get('support', [])])}")
            if resistance_count > 0:
                logging.info(f"📈 Resistance levels: {', '.join([f'${x:.2f}' for x in levels.get('resistance', [])])}")
                
        except Exception as level_error:
            logging.error(f"❌ Error detecting levels: {str(level_error)}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            st.error(f"❌ Error detecting levels: {str(level_error)}")
            levels = {"support": [], "resistance": []}

        # Fetch options data for options strategies
        options_data = None
        if analysis_type == "Options Trading Strategy":
            try:
                logging.info(f"🔄 Fetching options data for {ticker}...")
                options_start = time.time()
                from ..analysis.options_analysis import fetch_options_data
                
                # Try to get options data with a timeout
                options_data = fetch_options_data(ticker)
                options_time = time.time() - options_start
                
                if options_data:
                    iv_rank = options_data.get('iv_data', {}).get('iv_rank', 0)
                    iv_percentile = options_data.get('iv_data', {}).get('iv_percentile', 0)
                    expiry_count = len(options_data.get('expirations', []))
                    logging.info(f"✅ Options data fetched in {options_time:.2f}s - IV Rank: {iv_rank:.1f}%, {expiry_count} expirations available")
                    
                    # Store in session state for later use
                    try:
                        import streamlit as st
                        if 'options' not in st.session_state:
                            st.session_state['options'] = {}
                        st.session_state['options'][ticker] = options_data
                    except ImportError:
                        pass  # Not running in Streamlit
                else:
                    logging.warning(f"⚠️ No options data returned for {ticker}")
                
            except ImportError as import_error:
                logging.error(f"❌ Error importing options module: {import_error}")
                logging.info("ℹ️ This could be due to missing the fetch_options_data function or a dependency issue")
            except Exception as options_error:
                logging.error(f"❌ Error fetching options data: {str(options_error)}")
                logging.debug(f"Stack trace: {traceback.format_exc()}")
                st.error(f"❌ Error fetching options data: {str(options_error)}")
                traceback.print_exc()
                options_data = None
        
        total_time = time.time() - start_time
        logging.info(f"✅ Data pipeline completed in {total_time:.2f}s")
        logging.info(f"{'=' * 60}")
        
        # Return processed data, support/resistance levels and options data
        return data_with_indicators, levels, options_data
        
    except Exception as e:
        total_time = time.time() - start_time
        logging.error(f"❌ Error in data pipeline: {str(e)}")
        logging.error(f"❌ Pipeline failed after {total_time:.2f}s")
        logging.debug(f"Stack trace: {traceback.format_exc()}")
        st.error(f"❌ Error in data pipeline: {str(e)}")
        traceback.print_exc()
        return None, None, None
