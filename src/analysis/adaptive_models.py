"""
Adaptive model configurations optimized for different timeframes.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_adaptive_model_config(interval: str) -> dict:
    """Get model configuration optimized for specific timeframe."""
    
    if interval in ['1m', '5m']:  # High-frequency data
        config = {
            'RandomForest': {
                'n_estimators': 300,  # More trees for noise reduction
                'max_depth': 8,       # Shallower trees to prevent overfitting  
                'min_samples_split': 10,  # Higher to prevent overfitting
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42
            },
            'XGBoost': {
                'n_estimators': 200,
                'learning_rate': 0.01,  # Lower learning rate for stability
                'max_depth': 4,         # Shallower for high-frequency
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.2,       # Higher regularization
                'reg_lambda': 1.5,
                'objective': 'reg:squarederror',
                'random_state': 42
            },
            'CatBoost': {
                'iterations': 300,
                'learning_rate': 0.01,
                'depth': 4,
                'l2_leaf_reg': 8,       # Higher regularization
                'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU',
                'bootstrap_type': 'Bayesian'
            }
        }
        
    elif interval in ['15m', '30m']:  # Medium-frequency data  
        config = {
            'RandomForest': {
                'n_estimators': 400,
                'max_depth': 10,
                'min_samples_split': 8,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42
            },
            'XGBoost': {
                'n_estimators': 250,
                'learning_rate': 0.02,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.15,
                'reg_lambda': 1.2,
                'objective': 'reg:squarederror',
                'random_state': 42
            },
            'CatBoost': {
                'iterations': 400,
                'learning_rate': 0.02,
                'depth': 6,
                'l2_leaf_reg': 6,
                'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU',
                'bootstrap_type': 'Bayesian'
            }
        }
        
    elif interval in ['1h', '4h']:  # Hourly data
        config = {
            'RandomForest': {
                'n_estimators': 500,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42
            },
            'XGBoost': {
                'n_estimators': 300,
                'learning_rate': 0.03,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42
            },
            'CatBoost': {
                'iterations': 500,
                'learning_rate': 0.03,
                'depth': 8,
                'l2_leaf_reg': 5,
                'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU',
                'bootstrap_type': 'Bayesian'
            }
        }
        
    else:  # Daily or longer
        config = {
            'RandomForest': {
                'n_estimators': 500,
                'max_depth': 15,      # Deeper for longer timeframes
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42
            },
            'XGBoost': {
                'n_estimators': 400,
                'learning_rate': 0.04,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.05,    # Lower regularization for daily
                'reg_lambda': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            },
            'CatBoost': {
                'iterations': 500,
                'learning_rate': 0.03,
                'depth': 8,
                'l2_leaf_reg': 5,
                'random_seed': 42,
                'verbose': False,
                'task_type': 'CPU',
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.0
            }
        }
    
    logger.info(f"Using adaptive model configuration for {interval} interval")
    return config


def get_ensemble_weights_strategy(interval: str) -> str:
    """Get ensemble weighting strategy based on timeframe."""
    
    if interval in ['1m', '5m']:
        # For high-frequency data, prefer models that handle noise well
        return 'performance_weighted_with_noise_penalty'
    elif interval in ['15m', '30m', '1h']:
        # For medium frequency, balance performance and stability
        return 'performance_weighted_balanced'
    else:
        # For daily+, focus on raw performance
        return 'performance_weighted_simple'


def calculate_adaptive_ensemble_weights(cv_scores: dict, predictions: dict, interval: str) -> dict:
    """Calculate ensemble weights adapted to timeframe characteristics."""
    try:
        strategy = get_ensemble_weights_strategy(interval)
        weights = {}
        
        # Base weights from cross-validation performance
        base_weights = {}
        for name, scores in cv_scores.items():
            if isinstance(scores, list) and len(scores) > 0:
                # Use inverse of average RMSE as base weight
                avg_rmse = np.mean([score.get('rmse', float('inf')) if isinstance(score, dict) else score for score in scores])
                base_weights[name] = 1.0 / (avg_rmse + 1e-10)
            else:
                base_weights[name] = 0.1  # Small default weight
        
        # Apply strategy-specific adjustments
        if strategy == 'performance_weighted_with_noise_penalty':
            # Penalize complex models for high-frequency data
            model_complexity = {'RandomForest': 0.8, 'XGBoost': 1.0, 'CatBoost': 0.9}
            for name in base_weights:
                complexity_factor = model_complexity.get(name, 1.0)
                weights[name] = base_weights[name] * complexity_factor
                
        elif strategy == 'performance_weighted_balanced':
            # Balance performance with stability
            stability_bonus = {'CatBoost': 1.1, 'RandomForest': 1.05, 'XGBoost': 1.0}
            for name in base_weights:
                stability_factor = stability_bonus.get(name, 1.0)
                weights[name] = base_weights[name] * stability_factor
                
        else:  # performance_weighted_simple
            weights = base_weights.copy()
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        else:
            # Equal weights as fallback
            num_models = len(weights)
            for name in weights:
                weights[name] = 1.0 / num_models
        
        logger.info(f"Calculated adaptive ensemble weights using {strategy}: {weights}")
        return weights
        
    except Exception as e:
        logger.error(f"Error calculating adaptive ensemble weights: {str(e)}")
        # Equal weights fallback
        num_models = len(cv_scores)
        return {name: 1.0/num_models for name in cv_scores.keys()}