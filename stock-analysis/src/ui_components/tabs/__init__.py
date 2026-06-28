"""UI tab components for the trading analysis dashboard."""

from .technical_analysis import render_technical_analysis_tab
from .ai_recommendation import render_ai_recommendation_tab

__all__ = ['render_technical_analysis_tab', 'render_ai_recommendation_tab']