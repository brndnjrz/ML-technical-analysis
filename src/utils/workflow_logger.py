"""
Workflow Logger - Enhanced logging for step-by-step workflow monitoring

This module provides specialized logging functions to make workflow steps
more visible and trackable throughout the application.
"""

import logging
from datetime import datetime

def log_section_start(section_name):
    """
    Log the start of a major workflow section with clear visual separation
    
    Args:
        section_name: The name of the workflow section starting
    """
    logging.info(f"\n{'=' * 60}")
    logging.info(f"🚀 STARTING {section_name.upper()}")
    logging.info(f"{'=' * 60}")
    
def log_section_end(section_name):
    """
    Log the end of a major workflow section
    
    Args:
        section_name: The name of the workflow section ending
    """
    logging.info(f"\n{'=' * 60}")
    logging.info(f"✅ COMPLETED {section_name.upper()}")
    logging.info(f"{'=' * 60}\n")

def log_subsection_start(subsection_name):
    """
    Log the start of a workflow subsection
    
    Args:
        subsection_name: The name of the workflow subsection starting
    """
    logging.info(f"\n{'-' * 40}")
    logging.info(f"📌 {subsection_name}")
    logging.info(f"{'-' * 40}")
    
def log_subsection_end(subsection_name):
    """
    Log the end of a workflow subsection
    
    Args:
        subsection_name: The name of the workflow subsection ending
    """
    logging.info(f"{'-' * 40}")
    logging.info(f"✅ {subsection_name} COMPLETED")
    logging.info(f"{'-' * 40}\n")

def log_step(step_name, emoji="▶️"):
    """
    Log a single workflow step
    
    Args:
        step_name: The name/description of the step
        emoji: Optional emoji to prefix the step with
    """
    logging.info(f"{emoji} {step_name}")

def log_data_info(description, data):
    """
    Log information about data (shape, columns, etc.)
    
    Args:
        description: Description of the data
        data: The data object (DataFrame, dict, etc.)
    """
    if hasattr(data, 'shape'):
        logging.info(f"📊 {description} - Shape: {data.shape}")
        if hasattr(data, 'columns'):
            logging.info(f"   - Columns: {list(data.columns)[:5]}{'...' if len(data.columns) > 5 else ''}")
    elif isinstance(data, dict):
        logging.info(f"📊 {description} - Keys: {list(data.keys())[:5]}{'...' if len(data.keys()) > 5 else ''}")
    else:
        logging.info(f"📊 {description} - Type: {type(data).__name__}")

def log_prediction(ticker, price, confidence, regime=None):
    """
    Log a price prediction with formatted output
    
    Args:
        ticker: The stock ticker symbol
        price: The predicted price
        confidence: The confidence level (0-1)
        regime: Optional market regime classification
    """
    confidence_pct = confidence * 100 if confidence <= 1 else confidence
    
    logging.info(f"\n{'-' * 50}")
    logging.info(f"📈 PREDICTION FOR {ticker.upper()}:")
    logging.info(f"   - Predicted Price: ${price:.2f}")
    logging.info(f"   - Confidence: {confidence_pct:.1f}%")
    if regime:
        logging.info(f"   - Market Regime: {regime}")
    logging.info(f"{'-' * 50}\n")

def log_model_performance(model_name, metrics):
    """
    Log model performance metrics
    
    Args:
        model_name: The name of the model
        metrics: Dictionary of metrics
    """
    logging.info(f"🧠 MODEL {model_name.upper()} PERFORMANCE:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logging.info(f"   - {metric_name}: {metric_value:.4f}")
        else:
            logging.info(f"   - {metric_name}: {metric_value}")

def log_strategy_evaluation(strategy_name, scores):
    """
    Log strategy evaluation results
    
    Args:
        strategy_name: Name of the strategy
        scores: Dictionary of evaluation scores
    """
    logging.info(f"🎯 STRATEGY EVALUATION: {strategy_name}")
    for criterion, score in scores.items():
        if isinstance(score, float):
            logging.info(f"   - {criterion}: {score:.2f}")
        else:
            logging.info(f"   - {criterion}: {score}")

def log_timer_start(process_name):
    """
    Log the start of a timed process and return start time
    
    Args:
        process_name: Name of the process being timed
        
    Returns:
        start_time: The start time (datetime object)
    """
    start_time = datetime.now()
    logging.info(f"⏱️ STARTED: {process_name} at {start_time.strftime('%H:%M:%S')}")
    return start_time

def log_timer_end(process_name, start_time):
    """
    Log the end of a timed process with duration
    
    Args:
        process_name: Name of the process being timed
        start_time: The start time from log_timer_start
    """
    end_time = datetime.now()
    duration = end_time - start_time
    seconds = duration.total_seconds()
    
    if seconds < 60:
        duration_str = f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        duration_str = f"{minutes} min {remaining_seconds:.2f} sec"
    
    logging.info(f"⏱️ COMPLETED: {process_name} in {duration_str}")

def log_error(error_message, exception=None):
    """
    Log an error with formatted output
    
    Args:
        error_message: Description of what went wrong
        exception: Optional exception object
    """
    logging.error(f"❌ ERROR: {error_message}")
    if exception:
        logging.error(f"   - Exception: {str(exception)}")
        logging.debug(f"   - Exception type: {type(exception).__name__}")
