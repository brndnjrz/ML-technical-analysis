import pandas as pd
from typing import Dict, Any
import numpy as np
import logging

# Setup logger
logger = logging.getLogger(__name__)

class StrategyAgent:
    """
    Agent responsible for developing and optimizing trading strategies based on
    market analysis and historical performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def develop_strategy(self, analysis: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        logger.debug("💡 Developing trading strategy")
        strategy = {}
        try:
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(analysis)
            
            # Select appropriate strategy
            strategy = self._select_strategy(market_conditions, data)
            
            # Optimize parameters
            strategy = self._optimize_parameters(strategy, data)
            
            # Add risk management rules
            strategy = self._add_risk_rules(strategy, data)
            
        except Exception as e:
            print(f"Error developing strategy: {str(e)}")
            
        return strategy
    
    def _analyze_market_conditions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions to determine suitable strategies."""
        conditions = {
            'trend': None,
            'volatility': None,
            'momentum': None,
            'volume': None
        }
        
        try:
            # Determine trend
            if 'trend' in analysis:
                trend_data = analysis['trend']
                conditions['trend'] = {
                    'direction': trend_data.get('short_term'),
                    'strength': trend_data.get('strength_level')
                }
            
            # Assess volatility
            if 'volatility' in analysis:
                vol_data = analysis['volatility']
                conditions['volatility'] = vol_data.get('volatility_state')
            
            # Check momentum
            if 'momentum' in analysis:
                mom_data = analysis['momentum']
                conditions['momentum'] = {
                    'rsi_condition': mom_data.get('rsi', {}).get('condition'),
                    'macd_signal': mom_data.get('macd', {}).get('crossover')
                }
            
            # Analyze volume
            if 'volume' in analysis:
                vol_data = analysis['volume']
                conditions['volume'] = vol_data.get('volume_trend')
                
        except Exception as e:
            print(f"Error analyzing market conditions: {str(e)}")
            
        return conditions
    
    def _select_strategy(self, conditions: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Select the most appropriate trading strategy based on market conditions."""
        strategy = {
            'name': None,
            'type': None,
            'parameters': {},
            'confidence': 0.0
        }
        
        try:
            # Strategy selection logic based on market conditions
            if conditions['trend']['direction'] == 'bullish' and conditions['momentum']['macd_signal'] == 'bullish':
                strategy.update({
                    'name': 'Trend Following',
                    'type': 'long',
                    'parameters': {
                        'entry_condition': 'price_above_ma',
                        'exit_condition': 'price_below_ma',
                        'trailing_stop': True
                    },
                    'confidence': 0.8
                })
            elif conditions['trend']['direction'] == 'bearish' and conditions['momentum']['macd_signal'] == 'bearish':
                strategy.update({
                    'name': 'Trend Following',
                    'type': 'short',
                    'parameters': {
                        'entry_condition': 'price_below_ma',
                        'exit_condition': 'price_above_ma',
                        'trailing_stop': True
                    },
                    'confidence': 0.8
                })
            elif conditions['volatility'] == 'high':
                strategy.update({
                    'name': 'Mean Reversion',
                    'type': 'both',
                    'parameters': {
                        'entry_condition': 'bollinger_band',
                        'exit_condition': 'mean_touch',
                        'stop_loss': True
                    },
                    'confidence': 0.7
                })
            else:
                strategy.update({
                    'name': 'Range Trading',
                    'type': 'both',
                    'parameters': {
                        'entry_condition': 'support_resistance',
                        'exit_condition': 'target_reached',
                        'stop_loss': True
                    },
                    'confidence': 0.6
                })
                
        except Exception as e:
            print(f"Error selecting strategy: {str(e)}")
            
        return strategy
    
    def _optimize_parameters(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize strategy parameters based on historical data."""
        try:
            if strategy['name'] == 'Trend Following':
                strategy['parameters'].update({
                    'ma_period': self._optimize_ma_period(data),
                    'stop_loss': self._calculate_stop_loss(data),
                    'profit_target': self._calculate_profit_target(data)
                })
            elif strategy['name'] == 'Mean Reversion':
                strategy['parameters'].update({
                    'bb_period': self._optimize_bb_period(data),
                    'bb_std': self._optimize_bb_std(data),
                    'stop_loss': self._calculate_stop_loss(data)
                })
            elif strategy['name'] == 'Range Trading':
                strategy['parameters'].update({
                    'range_period': self._optimize_range_period(data),
                    'stop_loss': self._calculate_stop_loss(data)
                })
                
        except Exception as e:
            print(f"Error optimizing parameters: {str(e)}")
            
        return strategy
    
    def _add_risk_rules(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Add risk management rules to the strategy."""
        try:
            # Calculate volatility-based position sizing
            volatility = data['ATR'].iloc[-1]
            avg_volatility = data['ATR'].rolling(window=20).mean().iloc[-1]
            
            strategy['risk_management'] = {
                'position_size': self._calculate_position_size(volatility, avg_volatility),
                'max_loss_per_trade': 0.02,  # 2% maximum loss per trade
                'max_portfolio_risk': 0.06,  # 6% maximum portfolio risk
                'stop_loss': self._calculate_stop_loss(data),
                'trailing_stop': volatility * 2 if strategy['parameters'].get('trailing_stop') else None
            }
            
        except Exception as e:
            print(f"Error adding risk rules: {str(e)}")
            
        return strategy
    
    def _optimize_ma_period(self, data: pd.DataFrame) -> int:
        """Optimize moving average period."""
        return 20  # Implement optimization
    
    def _optimize_bb_period(self, data: pd.DataFrame) -> int:
        """Optimize Bollinger Bands period."""
        return 20  # Implement optimization
    
    def _optimize_bb_std(self, data: pd.DataFrame) -> float:
        """Optimize Bollinger Bands standard deviation."""
        return 2.0  # Implement optimization
    
    def _optimize_range_period(self, data: pd.DataFrame) -> int:
        """Optimize range period for range trading."""
        return 14  # Implement optimization
    
    def _calculate_stop_loss(self, data: pd.DataFrame) -> float:
        """Calculate appropriate stop loss level."""
        return data['ATR'].iloc[-1] * 2  # Simple ATR-based stop loss
    
    def _calculate_profit_target(self, data: pd.DataFrame) -> float:
        """Calculate appropriate profit target."""
        return data['ATR'].iloc[-1] * 3  # Simple ATR-based profit target
    
    def _calculate_position_size(self, current_volatility: float, avg_volatility: float) -> float:
        """Calculate appropriate position size based on volatility."""
        if current_volatility > avg_volatility * 1.5:
            return 0.5  # Reduce position size in high volatility
        elif current_volatility < avg_volatility * 0.5:
            return 1.0  # Full position size in low volatility
        else:
            return 0.75  # Normal position size
