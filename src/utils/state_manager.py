"""Streamlit session state management utilities."""
import streamlit as st
from .app_config import SessionKeys


class AppStateManager:
    """Manage Streamlit session state for the application."""
    
    @staticmethod
    def get_stock_data():
        """Get stock data from session state."""
        return st.session_state.get(SessionKeys.STOCK_DATA)
    
    @staticmethod
    def set_stock_data(data):
        """Set stock data in session state."""
        st.session_state[SessionKeys.STOCK_DATA] = data
    
    @staticmethod
    def get_levels():
        """Get support/resistance levels from session state."""
        levels = st.session_state.get(SessionKeys.LEVELS, {})
        if not isinstance(levels, dict) or not ("support" in levels and "resistance" in levels):
            return {'support': [], 'resistance': []}
        return levels
    
    @staticmethod
    def set_levels(levels):
        """Set support/resistance levels in session state."""
        st.session_state[SessionKeys.LEVELS] = levels
    
    @staticmethod
    def get_options_data():
        """Get options data from session state."""
        return st.session_state.get(SessionKeys.OPTIONS_DATA, {})
    
    @staticmethod
    def set_options_data(options_data):
        """Set options data in session state."""
        st.session_state[SessionKeys.OPTIONS_DATA] = options_data
    
    @staticmethod
    def get_active_indicators():
        """Get active indicators from session state."""
        return st.session_state.get(SessionKeys.ACTIVE_INDICATORS, [])
    
    @staticmethod
    def set_active_indicators(indicators):
        """Set active indicators in session state."""
        st.session_state[SessionKeys.ACTIVE_INDICATORS] = indicators
    
    @staticmethod
    def get_analysis_type():
        """Get analysis type from session state."""
        return st.session_state.get(SessionKeys.ANALYSIS_TYPE)
    
    @staticmethod
    def set_analysis_type(analysis_type):
        """Set analysis type in session state."""
        st.session_state[SessionKeys.ANALYSIS_TYPE] = analysis_type
    
    @staticmethod
    def get_ai_analysis_result():
        """Get AI analysis result from session state."""
        return st.session_state.get(SessionKeys.AI_ANALYSIS_RESULT)
    
    @staticmethod
    def set_ai_analysis_result(analysis, chart_path, recommendation):
        """Set AI analysis result in session state."""
        st.session_state[SessionKeys.AI_ANALYSIS_RESULT] = (analysis, chart_path, recommendation)
    
    @staticmethod
    def clear_ai_analysis_result():
        """Clear AI analysis result from session state."""
        if SessionKeys.AI_ANALYSIS_RESULT in st.session_state:
            del st.session_state[SessionKeys.AI_ANALYSIS_RESULT]

    @staticmethod
    def should_run_analysis():
        """Check if analysis should be run."""
        return st.session_state.get(SessionKeys.RUN_ANALYSIS, False)
    
    @staticmethod
    def set_run_analysis(should_run):
        """Set whether analysis should be run."""
        st.session_state[SessionKeys.RUN_ANALYSIS] = should_run
    
    @staticmethod
    def has_stock_data():
        """Check if stock data exists in session state."""
        return SessionKeys.STOCK_DATA in st.session_state
    
    @staticmethod
    def clear_all_analysis_state():
        """Clear all analysis-related state."""
        keys_to_clear = [
            SessionKeys.AI_ANALYSIS_RESULT,
            SessionKeys.RUN_ANALYSIS
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]