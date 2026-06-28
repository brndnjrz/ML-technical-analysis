"""
Enhanced cross-validation strategies adapted to different timeframes.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def get_adaptive_cv_strategy(data_length: int, interval: str):
    """Get cross-validation strategy adapted to data size and timeframe."""
    
    if interval in ['1m', '5m']:  # High-frequency data
        # More splits for high-frequency data to capture patterns
        min_splits = max(3, min(10, data_length // 50))
        max_train_size = min(data_length//2, 1000)  # Limit training size for speed
        return TimeSeriesSplit(n_splits=min_splits, max_train_size=max_train_size)
        
    elif interval in ['15m', '30m']:  # Medium-frequency
        min_splits = max(3, min(8, data_length // 30))
        max_train_size = data_length//3
        return TimeSeriesSplit(n_splits=min_splits, max_train_size=max_train_size)
        
    elif interval in ['1h', '4h']:  # Hourly
        min_splits = max(3, min(7, data_length // 20))
        return TimeSeriesSplit(n_splits=min_splits)
        
    else:  # Daily or longer
        min_splits = max(3, min(5, data_length // 15))
        return TimeSeriesSplit(n_splits=min_splits)


def calculate_timeframe_specific_metrics(y_true, y_pred, interval: str) -> dict:
    """Calculate metrics that are most relevant for specific timeframes."""
    
    metrics = {}
    
    # Base metrics for all timeframes
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Timeframe-specific metrics
    if interval in ['1m', '5m', '15m', '30m']:
        # For short timeframes, focus on directional accuracy
        directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y_true))
        metrics['directional_accuracy'] = directional_accuracy
        
        # Hit rate (what percentage of predictions are within reasonable range)
        threshold = np.std(y_true) * 0.5  # Within 0.5 standard deviations
        hit_rate = np.mean(np.abs(y_pred - y_true) <= threshold)
        metrics['hit_rate'] = hit_rate
        
        # Sharpe-like ratio for trading performance
        if np.std(y_pred) > 0:
            prediction_sharpe = np.mean(y_pred) / np.std(y_pred)
            metrics['prediction_sharpe'] = prediction_sharpe
        else:
            metrics['prediction_sharpe'] = 0
            
    else:
        # For longer timeframes, focus on absolute accuracy
        if np.mean(np.abs(y_true)) > 0:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = 100
        
        # R-squared for longer timeframes
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        metrics['r2'] = r2
    
    return metrics


def validate_prediction_performance(models: dict, X_train: pd.DataFrame, y_train: pd.Series, 
                                  cv_strategy, interval: str) -> dict:
    """Enhanced validation with timeframe-specific metrics."""
    
    validation_results = {}
    
    try:
        for name, model in models.items():
            logger.info(f"Validating {name} model...")
            scores = []
            
            fold_count = 0
            for train_idx, val_idx in cv_strategy.split(X_train):
                fold_count += 1
                
                try:
                    X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                    X_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                    y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                    y_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                    
                    # Skip if training set is too small
                    if len(X_tr) < 5:
                        continue
                    
                    # Train model
                    model.fit(X_tr, y_tr)
                    predictions = model.predict(X_val)
                    
                    # Calculate timeframe-specific metrics
                    fold_metrics = calculate_timeframe_specific_metrics(y_val, predictions, interval)
                    scores.append(fold_metrics)
                    
                except Exception as fold_error:
                    logger.warning(f"Error in fold {fold_count} for {name}: {str(fold_error)}")
                    continue
            
            if scores:
                validation_results[name] = scores
                
                # Log average performance
                avg_metrics = {}
                for metric in scores[0].keys():
                    avg_metrics[metric] = np.mean([score[metric] for score in scores])
                logger.info(f"{name} average metrics: {avg_metrics}")
            else:
                logger.warning(f"No valid scores for {name}")
                validation_results[name] = []
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in validation: {str(e)}")
        return {}


def select_best_models(validation_results: dict, interval: str, max_models: int = 3) -> list:
    """Select best performing models based on timeframe-specific criteria."""
    
    try:
        model_scores = {}
        
        # Define primary metric based on timeframe
        if interval in ['1m', '5m', '15m', '30m']:
            primary_metric = 'directional_accuracy'
            higher_is_better = True
        else:
            primary_metric = 'rmse'
            higher_is_better = False
        
        # Calculate average primary metric for each model
        for name, scores in validation_results.items():
            if scores:
                avg_score = np.mean([score.get(primary_metric, 0 if higher_is_better else float('inf')) 
                                   for score in scores])
                model_scores[name] = avg_score
        
        # Sort models by performance
        if higher_is_better:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
        
        # Select top models
        best_models = [name for name, score in sorted_models[:max_models]]
        
        logger.info(f"Selected best models for {interval}: {best_models}")
        return best_models
        
    except Exception as e:
        logger.error(f"Error selecting best models: {str(e)}")
        return list(validation_results.keys())[:max_models]