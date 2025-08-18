"""
Metrics tracking system for AI accuracy optimization.
Tracks directional accuracy, range predictions, strategy performance, and calibration.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AccuracyMetrics:
    """Track and analyze AI prediction accuracy across multiple dimensions"""
    
    def __init__(self, metrics_file: str = "metrics/accuracy_log.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(self, 
                      ticker: str, 
                      recommendation: Dict[str, Any], 
                      regime: str,
                      market_data: Dict[str, Any],
                      vision_enabled: bool = False,
                      prompt_version: str = "vA") -> str:
        """
        Log a prediction for future accuracy measurement
        
        Returns:
            str: Prediction ID for later outcome tracking
        """
        try:
            prediction_id = f"{ticker}_{int(datetime.now().timestamp())}"
            
            # Extract key prediction components
            action = recommendation.get('action', 'HOLD')
            confidence = recommendation.get('strategy', {}).get('confidence', 0.5)
            up_prob = self._extract_up_probability(recommendation, action, confidence)
            
            # Extract market context
            current_price = market_data.get('current_price', 0)
            rsi = market_data.get('RSI', 50)
            atr = market_data.get('ATR', 0)
            ivr = market_data.get('iv_rank', 0)
            adx = market_data.get('ADX', 25)
            
            # Range prediction (inside probability)
            support_levels = recommendation.get('vision_analysis', {}).get('support', [])
            resistance_levels = recommendation.get('vision_analysis', {}).get('resistance', [])
            
            if support_levels and resistance_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.98)
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.02)
                range_width_pct = (nearest_resistance - nearest_support) / current_price
                inside_prob = 0.7 if range_width_pct > 0.04 else 0.5  # Wide range = higher inside probability
            else:
                inside_prob = 0.5
                nearest_support = current_price * 0.98
                nearest_resistance = current_price * 1.02
            
            # Strategy details
            strategy_name = recommendation.get('strategy', {}).get('name', 'Unknown')
            
            # Risk assessment
            risk_level = recommendation.get('risk_assessment', {}).get('risk_level', 'MEDIUM')
            
            prediction_record = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'regime': regime,
                'prompt_version': prompt_version,
                'vision_enabled': vision_enabled,
                
                # Market context at prediction time
                'current_price': current_price,
                'rsi': rsi,
                'atr': atr,
                'atr_pct': atr / current_price if current_price > 0 else 0,
                'ivr': ivr,
                'adx': adx,
                
                # Predictions
                'action': action,
                'confidence': confidence,
                'up_prob': up_prob,
                'inside_prob': inside_prob,
                'strategy_name': strategy_name,
                'risk_level': risk_level,
                
                # Range bounds for range accuracy
                'support_level': nearest_support,
                'resistance_level': nearest_resistance,
                'range_width_pct': (nearest_resistance - nearest_support) / current_price if current_price > 0 else 0,
                
                # Parameters for strategy evaluation
                'entry_price': recommendation.get('parameters', {}).get('entry_price', current_price),
                'stop_loss': recommendation.get('parameters', {}).get('stop_loss'),
                'take_profit': recommendation.get('parameters', {}).get('take_profit'),
                
                # Outcome fields (to be filled later)
                'outcome_recorded': False,
                'directional_correct_1d': None,
                'directional_correct_3d': None,
                'directional_correct_7d': None,
                'inside_range_1d': None,
                'inside_range_3d': None,
                'inside_range_7d': None,
                'max_favorable_move_pct': None,
                'max_adverse_move_pct': None,
                'realized_pnl_pct': None,
                'stop_loss_hit': None
            }
            
            # Log to file
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(prediction_record) + '\n')
                
            logger.info(f"📊 Logged prediction {prediction_id}: {action} {ticker} (confidence: {confidence:.2f})")
            return prediction_id
            
        except Exception as e:
            logger.error(f"❌ Error logging prediction: {e}")
            return ""
    
    def record_outcome(self, prediction_id: str, outcome_data: Dict[str, Any]) -> bool:
        """Record the actual outcome for a prediction"""
        try:
            # Read all predictions
            predictions = self._read_predictions()
            
            # Find and update the prediction
            updated = False
            for pred in predictions:
                if pred.get('prediction_id') == prediction_id:
                    pred.update(outcome_data)
                    pred['outcome_recorded'] = True
                    pred['outcome_timestamp'] = datetime.now().isoformat()
                    updated = True
                    break
            
            if updated:
                # Rewrite the file
                with open(self.metrics_file, 'w') as f:
                    for pred in predictions:
                        f.write(json.dumps(pred) + '\n')
                logger.info(f"✅ Recorded outcome for {prediction_id}")
                return True
            else:
                logger.warning(f"⚠️ Prediction {prediction_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error recording outcome: {e}")
            return False
    
    def calculate_accuracy_metrics(self, days_back: int = 30, by_regime: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive accuracy metrics"""
        try:
            predictions = self._read_predictions()
            
            # Filter recent predictions with outcomes
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_predictions = [
                p for p in predictions 
                if datetime.fromisoformat(p.get('timestamp', '1970-01-01')) >= cutoff_date
                and p.get('outcome_recorded', False)
            ]
            
            if not recent_predictions:
                return {'error': 'No predictions with outcomes found'}
            
            # Overall metrics
            metrics = {
                'total_predictions': len(recent_predictions),
                'evaluation_period_days': days_back,
                'directional_accuracy': self._calculate_directional_accuracy(recent_predictions),
                'range_accuracy': self._calculate_range_accuracy(recent_predictions),
                'calibration': self._calculate_calibration(recent_predictions),
                'risk_accuracy': self._calculate_risk_accuracy(recent_predictions),
                'strategy_performance': self._calculate_strategy_performance(recent_predictions)
            }
            
            # By regime breakdown
            if by_regime:
                regimes = set(p.get('regime', 'unknown') for p in recent_predictions)
                metrics['by_regime'] = {}
                
                for regime in regimes:
                    regime_preds = [p for p in recent_predictions if p.get('regime') == regime]
                    if regime_preds:
                        metrics['by_regime'][regime] = {
                            'count': len(regime_preds),
                            'directional_accuracy': self._calculate_directional_accuracy(regime_preds),
                            'range_accuracy': self._calculate_range_accuracy(regime_preds),
                            'calibration': self._calculate_calibration(regime_preds)
                        }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {'error': str(e)}
    
    def _extract_up_probability(self, recommendation: Dict, action: str, confidence: float) -> float:
        """Convert action + confidence to directional probability"""
        if action == 'BUY':
            return 0.5 + (confidence * 0.4)  # 0.5 to 0.9 range
        elif action == 'SELL':
            return 0.5 - (confidence * 0.4)  # 0.1 to 0.5 range
        else:  # HOLD
            return 0.5
    
    def _calculate_directional_accuracy(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate hit rates for different time horizons"""
        horizons = ['1d', '3d', '7d']
        accuracy = {}
        
        for horizon in horizons:
            field = f'directional_correct_{horizon}'
            correct_predictions = [p for p in predictions if p.get(field) is not None]
            
            if correct_predictions:
                hit_rate = sum(1 for p in correct_predictions if p[field]) / len(correct_predictions)
                accuracy[f'{horizon}_hit_rate'] = hit_rate
                accuracy[f'{horizon}_count'] = len(correct_predictions)
            else:
                accuracy[f'{horizon}_hit_rate'] = 0.0
                accuracy[f'{horizon}_count'] = 0
        
        return accuracy
    
    def _calculate_range_accuracy(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate range prediction accuracy"""
        horizons = ['1d', '3d', '7d']
        accuracy = {}
        
        for horizon in horizons:
            field = f'inside_range_{horizon}'
            range_predictions = [p for p in predictions if p.get(field) is not None]
            
            if range_predictions:
                inside_rate = sum(1 for p in range_predictions if p[field]) / len(range_predictions)
                accuracy[f'{horizon}_inside_rate'] = inside_rate
                accuracy[f'{horizon}_count'] = len(range_predictions)
            else:
                accuracy[f'{horizon}_inside_rate'] = 0.0
                accuracy[f'{horizon}_count'] = 0
        
        return accuracy
    
    def _calculate_calibration(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate calibration metrics (Brier score, reliability)"""
        # Get predictions with confidence and 7-day outcomes
        calibration_data = [
            (p.get('confidence', 0.5), 1 if p.get('directional_correct_7d', False) else 0)
            for p in predictions 
            if p.get('directional_correct_7d') is not None
        ]
        
        if not calibration_data:
            return {'brier_score': 1.0, 'reliability': 0.0}
        
        confidences, outcomes = zip(*calibration_data)
        
        # Brier score (lower is better, 0-1 range)
        brier_score = np.mean([(conf - outcome) ** 2 for conf, outcome in calibration_data])
        
        # Reliability score (how well confidence matches actual performance)
        # Bin predictions by confidence level
        bins = np.linspace(0, 1, 6)  # 5 bins: 0-0.2, 0.2-0.4, etc.
        reliability_scores = []
        
        for i in range(len(bins) - 1):
            bin_mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
            if np.sum(bin_mask) > 0:
                bin_confidences = np.array(confidences)[bin_mask]
                bin_outcomes = np.array(outcomes)[bin_mask]
                
                avg_confidence = np.mean(bin_confidences)
                actual_rate = np.mean(bin_outcomes)
                
                # Add weighted reliability contribution
                weight = np.sum(bin_mask) / len(calibration_data)
                reliability_scores.append(weight * abs(avg_confidence - actual_rate))
        
        reliability = 1.0 - sum(reliability_scores) if reliability_scores else 0.5
        
        return {
            'brier_score': brier_score,
            'reliability': max(0.0, reliability),
            'calibration_samples': len(calibration_data)
        }
    
    def _calculate_risk_accuracy(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate how well risk assessments match actual volatility"""
        risk_data = [
            (p.get('risk_level', 'MEDIUM'), p.get('max_adverse_move_pct', 0))
            for p in predictions 
            if p.get('max_adverse_move_pct') is not None
        ]
        
        if not risk_data:
            return {'accuracy': 0.0}
        
        # Define risk thresholds
        risk_thresholds = {'LOW': 0.02, 'MEDIUM': 0.05, 'HIGH': 0.10}
        correct_risk_assessments = 0
        
        for risk_level, actual_move in risk_data:
            threshold = risk_thresholds.get(risk_level, 0.05)
            
            # Check if actual move aligns with predicted risk
            if risk_level == 'LOW' and abs(actual_move) <= threshold:
                correct_risk_assessments += 1
            elif risk_level == 'MEDIUM' and threshold * 0.5 < abs(actual_move) <= threshold * 2:
                correct_risk_assessments += 1
            elif risk_level == 'HIGH' and abs(actual_move) > threshold * 0.5:
                correct_risk_assessments += 1
        
        return {
            'accuracy': correct_risk_assessments / len(risk_data),
            'samples': len(risk_data)
        }
    
    def _calculate_strategy_performance(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate performance by strategy type"""
        strategies = {}
        
        for pred in predictions:
            strategy = pred.get('strategy_name', 'Unknown')
            if strategy not in strategies:
                strategies[strategy] = {
                    'count': 0,
                    'hit_rate_7d': 0,
                    'avg_pnl': 0,
                    'predictions': []
                }
            
            strategies[strategy]['count'] += 1
            strategies[strategy]['predictions'].append(pred)
            
            if pred.get('directional_correct_7d') is not None:
                strategies[strategy]['hit_rate_7d'] += 1 if pred['directional_correct_7d'] else 0
            
            if pred.get('realized_pnl_pct') is not None:
                strategies[strategy]['avg_pnl'] += pred['realized_pnl_pct']
        
        # Calculate averages
        for strategy, data in strategies.items():
            count = data['count']
            if count > 0:
                data['hit_rate_7d'] = data['hit_rate_7d'] / count
                data['avg_pnl'] = data['avg_pnl'] / count
            
            # Remove raw predictions to keep output clean
            del data['predictions']
        
        return strategies
    
    def _read_predictions(self) -> List[Dict[str, Any]]:
        """Read all predictions from the metrics file"""
        predictions = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            predictions.append(json.loads(line))
            except Exception as e:
                logger.error(f"❌ Error reading predictions: {e}")
        
        return predictions

# Global metrics instance
metrics_tracker = AccuracyMetrics()

def log_prediction(ticker: str, recommendation: Dict[str, Any], regime: str, 
                  market_data: Dict[str, Any], **kwargs) -> str:
    """Convenience function for logging predictions"""
    return metrics_tracker.log_prediction(ticker, recommendation, regime, market_data, **kwargs)

def record_outcome(prediction_id: str, outcome_data: Dict[str, Any]) -> bool:
    """Convenience function for recording outcomes"""
    return metrics_tracker.record_outcome(prediction_id, outcome_data)

def get_accuracy_report(days_back: int = 30) -> Dict[str, Any]:
    """Convenience function for getting accuracy metrics"""
    return metrics_tracker.calculate_accuracy_metrics(days_back)
