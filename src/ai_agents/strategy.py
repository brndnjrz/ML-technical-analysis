import pandas as pd
from typing import Dict, Any
import numpy as np
import logging
from ..trading_strategies import strategies_data

# Setup logger
logger = logging.getLogger(__name__)

class StrategyAgent:
    """
    Agent responsible for developing and optimizing trading strategies based on
    market analysis and historical performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Load trading strategies data for intelligent strategy selection
        self.strategies_db = {strategy['Strategy']: strategy for strategy in strategies_data}
        self.available_strategies = list(self.strategies_db.keys())
        
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
            # Enhanced strategy selection using trading_strategies.py data
            selected_strategy_info = self._match_strategy_to_conditions(conditions, data)
            
            if selected_strategy_info:
                strategy_data = self.strategies_db[selected_strategy_info['name']]
                
                strategy.update({
                    'name': selected_strategy_info['name'],
                    'type': selected_strategy_info['type'],
                    'description': strategy_data['Description'],
                    'timeframe': strategy_data['Timeframe'],
                    'pros': strategy_data['Pros'],
                    'cons': strategy_data['Cons'],
                    'when_to_use': strategy_data['When to Use'],
                    'suitable_for': strategy_data['Suitable For'],
                    'parameters': self._extract_strategy_parameters(strategy_data, conditions),
                    'confidence': selected_strategy_info['confidence']
                })
            else:
                # Fallback to original logic
                strategy = self._fallback_strategy_selection(conditions, data)
                
        except Exception as e:
            print(f"Error selecting strategy: {str(e)}")
            strategy = self._fallback_strategy_selection(conditions, data)
            
        return strategy
    
    def _match_strategy_to_conditions(self, conditions: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Match current market conditions to appropriate trading strategies."""
        strategy_matches = []
        
        try:
            # Calculate market volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Get trend strength
            trend_direction = conditions.get('trend', {}).get('direction', 'neutral')
            trend_strength = conditions.get('trend', {}).get('strength', 'weak')
            momentum_condition = conditions.get('momentum', {}).get('rsi_condition', 'neutral')
            volume_trend = conditions.get('volume', 'normal')
            
            # Strategy matching logic based on market conditions
            if volatility > 0.3 and momentum_condition in ['overbought', 'oversold']:
                # High volatility with extreme momentum - good for straddle/strangle
                strategy_matches.append({
                    'name': 'Straddle/Strangle',
                    'type': 'both',
                    'confidence': 0.85,
                    'reason': 'High volatility with extreme momentum levels'
                })
            
            if trend_direction in ['bullish', 'bearish'] and trend_strength == 'strong':
                if volume_trend == 'above_average':
                    # Strong trend with volume confirmation - good for trend following
                    if len(data) <= 50:  # Short-term data
                        strategy_matches.append({
                            'name': 'Day Trading Calls/Puts',
                            'type': 'long' if trend_direction == 'bullish' else 'short',
                            'confidence': 0.8,
                            'reason': 'Strong trend with volume confirmation in short timeframe'
                        })
                    else:  # Longer-term data
                        strategy_matches.append({
                            'name': 'Swing Trading',
                            'type': 'long' if trend_direction == 'bullish' else 'short',
                            'confidence': 0.75,
                            'reason': 'Strong trend suitable for swing trading'
                        })
            
            if volatility < 0.15 and momentum_condition == 'neutral':
                # Low volatility, neutral momentum - good for range strategies
                strategy_matches.append({
                    'name': 'Iron Condor',
                    'type': 'neutral',
                    'confidence': 0.7,
                    'reason': 'Low volatility environment suitable for premium selling'
                })
                
                strategy_matches.append({
                    'name': 'Butterfly Spread',
                    'type': 'neutral',
                    'confidence': 0.65,
                    'reason': 'Stable market conditions for range-bound strategy'
                })
            
            if trend_direction == 'bullish' and data['Close'].iloc[-1] > data.get('SMA_50', data['Close']).iloc[-1]:
                # Bullish trend above major moving average
                strategy_matches.append({
                    'name': 'Covered Calls',
                    'type': 'long',
                    'confidence': 0.7,
                    'reason': 'Bullish trend suitable for income generation'
                })
            
            if volatility > 0.25:
                # Higher volatility - protective strategies
                strategy_matches.append({
                    'name': 'Protective Puts',
                    'type': 'protective',
                    'confidence': 0.6,
                    'reason': 'High volatility warrants portfolio protection'
                })
            
            # Return the highest confidence match
            if strategy_matches:
                return max(strategy_matches, key=lambda x: x['confidence'])
            
        except Exception as e:
            print(f"Error matching strategy to conditions: {str(e)}")
        
        return None
    
    def _extract_strategy_parameters(self, strategy_data: Dict[str, Any], conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and customize strategy parameters based on market conditions."""
        parameters = {}
        
        try:
            strategy_name = strategy_data['Strategy']
            
            # Base parameters from strategy timeframe
            timeframe_info = strategy_data.get('Timeframe Used With', '')
            if 'days' in timeframe_info.lower() or 'weeks' in timeframe_info.lower():
                parameters['holding_period'] = 'medium_term'
            elif 'minutes' in timeframe_info.lower() or 'hours' in timeframe_info.lower():
                parameters['holding_period'] = 'short_term'
            else:
                parameters['holding_period'] = 'flexible'
            
            # Strategy-specific parameters
            if strategy_name == 'Day Trading Calls/Puts':
                parameters.update({
                    'entry_condition': 'momentum_breakout',
                    'exit_condition': 'same_day_close',
                    'stop_loss': 'tight_atr_based',
                    'profit_target': 'quick_scalp',
                    'max_holding_time': '6_hours'
                })
            
            elif strategy_name == 'Iron Condor':
                parameters.update({
                    'entry_condition': 'high_iv_rank',
                    'exit_condition': 'profit_target_or_time',
                    'strike_selection': 'otm_balanced',
                    'profit_target': '25_percent_max_profit',
                    'time_decay_benefit': True
                })
            
            elif strategy_name == 'Straddle/Strangle':
                parameters.update({
                    'entry_condition': 'volatility_expansion_expected',
                    'exit_condition': 'volatility_crush_or_profit',
                    'strike_selection': 'atm_or_otm',
                    'profit_target': 'volatility_based',
                    'time_decay_risk': True
                })
            
            elif strategy_name == 'Swing Trading':
                parameters.update({
                    'entry_condition': 'trend_continuation',
                    'exit_condition': 'trend_reversal_or_target',
                    'stop_loss': 'swing_low_high',
                    'profit_target': 'resistance_support_levels',
                    'trailing_stop': True
                })
            
            elif strategy_name == 'Covered Calls':
                parameters.update({
                    'entry_condition': 'own_underlying_stock',
                    'exit_condition': 'expiration_or_buyback',
                    'strike_selection': 'otm_above_resistance',
                    'income_strategy': True,
                    'assignment_risk': True
                })
            
            # Add market condition adjustments
            if conditions.get('volatility') == 'high':
                parameters['position_size_adjustment'] = 'reduce_size'
                parameters['risk_management'] = 'enhanced'
            
            if conditions.get('trend', {}).get('strength') == 'strong':
                parameters['confidence_boost'] = True
                parameters['trend_following_bias'] = True
                
        except Exception as e:
            print(f"Error extracting strategy parameters: {str(e)}")
        
        return parameters
    
    def _fallback_strategy_selection(self, conditions: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to original strategy selection logic."""
        strategy = {
            'name': None,
            'type': None,
            'parameters': {},
            'confidence': 0.0
        }
        
        try:
            # Original strategy selection logic
            if conditions.get('trend', {}).get('direction') == 'bullish' and conditions.get('momentum', {}).get('macd_signal') == 'bullish':
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
            elif conditions.get('trend', {}).get('direction') == 'bearish' and conditions.get('momentum', {}).get('macd_signal') == 'bearish':
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
            elif conditions.get('volatility') == 'high':
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
            print(f"Error in fallback strategy selection: {str(e)}")
            
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
