"""
Analysis Module
==============
Contains all analysis functionality including AI analysis, indicators, predictions, 
and adaptive feature engineering for improved accuracy across different timeframes.
"""

from .indicators import calculate_indicators, detect_support_resistance
from .prediction import predict_next_period_close, get_fundamental_metrics
from .ai_analysis import run_ai_analysis
from .market_regime import detect_market_regime, generate_regime_specific_features
from .adaptive_features import engineer_adaptive_features, get_adaptive_periods
from .preprocessing import create_preprocessing_pipeline, create_target_variable
from .adaptive_models import get_adaptive_model_config, calculate_adaptive_ensemble_weights
from .enhanced_validation import get_adaptive_cv_strategy, validate_prediction_performance

__all__ = [
    'calculate_indicators',
    'detect_support_resistance',
    'predict_next_period_close',
    'get_fundamental_metrics',
    'run_ai_analysis',
    'detect_market_regime',
    'generate_regime_specific_features',
    'engineer_adaptive_features',
    'get_adaptive_periods',
    'create_preprocessing_pipeline',
    'create_target_variable',
    'get_adaptive_model_config',
    'calculate_adaptive_ensemble_weights',
    'get_adaptive_cv_strategy',
    'validate_prediction_performance'
]
