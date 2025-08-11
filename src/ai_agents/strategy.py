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
        
    def develop_strategy(self, analysis: Dict[str, Any], data: pd.DataFrame, options_priority: bool = False, options_data: Dict = None) -> Dict[str, Any]:
        logger.debug("💡 Developing trading strategy")
        strategy = {}
        try:
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(analysis)
            
            # Enhance conditions with options-specific information if available
            if options_data and options_priority:
                market_conditions = self._enrich_conditions_with_options_data(market_conditions, options_data, data)
                logger.info("📊 Enhanced market conditions with options data")
            
            # Select appropriate strategy
            strategy = self._select_strategy(market_conditions, data, options_priority)
            
            # Optimize parameters
            strategy = self._optimize_parameters(strategy, data)
            
            # Add risk management rules
            strategy = self._add_risk_rules(strategy, data)
            
        except Exception as e:
            print(f"Error developing strategy: {str(e)}")
            
        return strategy
        
    def _enrich_conditions_with_options_data(self, conditions: Dict[str, Any], options_data: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhance market conditions assessment with options data."""
        try:
            # Add options-specific conditions to the market assessment
            if not 'options' in conditions:
                conditions['options'] = {}
                
            # Get implied volatility if available
            if 'implied_volatility' in options_data:
                conditions['options']['iv'] = options_data['implied_volatility']
                
                # Calculate IV rank/percentile if historical data is available
                if 'iv_history' in options_data and len(options_data['iv_history']) > 0:
                    iv_history = options_data['iv_history']
                    current_iv = options_data['implied_volatility']
                    iv_rank = (current_iv - min(iv_history)) / (max(iv_history) - min(iv_history)) if max(iv_history) > min(iv_history) else 0.5
                    conditions['options']['iv_rank'] = iv_rank
                    
                    # Determine if IV is high, low, or normal
                    if iv_rank > 0.7:
                        conditions['options']['iv_state'] = 'high'
                        conditions['options']['recommended_strategies'] = ['Iron Condor', 'Credit Spread', 'Calendar Spread']
                    elif iv_rank < 0.3:
                        conditions['options']['iv_state'] = 'low'
                        conditions['options']['recommended_strategies'] = ['Long Call/Put', 'Debit Spread', 'LEAPS']
                    else:
                        conditions['options']['iv_state'] = 'normal'
                        conditions['options']['recommended_strategies'] = ['Vertical Spread', 'Butterfly', 'Diagonal Spread']
            
            # Check for special conditions in the data if they exist
            if 'IC_SUITABILITY' in data.columns:
                ic_score = data['IC_SUITABILITY'].iloc[-1]
                conditions['options']['ic_suitability'] = ic_score
                
            # Check for volatility skew
            if 'VOL_SKEW' in data.columns:
                vol_skew = data['VOL_SKEW'].iloc[-1]
                conditions['options']['volatility_skew'] = vol_skew
                
                # Recommend strategies based on volatility skew
                if vol_skew > 1.2:  # Significant downside protection premium
                    conditions['options']['skew_bias'] = 'downside_protection'
                    conditions['options']['skew_strategies'] = ['Put Credit Spread', 'Put Ratio Spread']
                elif vol_skew < 0.8:  # Upside calls relatively cheap
                    conditions['options']['skew_bias'] = 'upside_potential'
                    conditions['options']['skew_strategies'] = ['Call Debit Spread', 'Call Backspread']
                else:
                    conditions['options']['skew_bias'] = 'balanced'
            
            logger.info(f"📊 Added options data to market conditions assessment")
            
        except Exception as e:
            logger.warning(f"Error enriching conditions with options data: {str(e)}")
            
        return conditions
    
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
    
    def _select_strategy(self, conditions: Dict[str, Any], data: pd.DataFrame, options_priority: bool = False) -> Dict[str, Any]:
        """Select the most appropriate trading strategy based on market conditions."""
        strategy = {
            'name': None,
            'type': None,
            'parameters': {},
            'confidence': 0.0
        }
        
        try:
            # Enhanced strategy selection using trading_strategies.py data
            selected_strategy_info = self._match_strategy_to_conditions(conditions, data, options_priority)
            
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
    
    def _match_strategy_to_conditions(self, conditions: Dict[str, Any], data: pd.DataFrame, options_priority: bool = False) -> Dict[str, Any]:
        """Match current market conditions to appropriate trading strategies.
        
        Focus on a limited set of strategies:
        - Day Trading Calls/Puts
        - Iron Condor
        - Swing Trading
        """
        strategy_matches = []
        
        # Get trend strength
        trend_direction = conditions.get('trend', {}).get('direction', 'neutral')
        trend_strength = conditions.get('trend', {}).get('strength', 'weak')
        momentum_condition = conditions.get('momentum', {}).get('rsi_condition', 'neutral')
        volume_trend = conditions.get('volume', 'normal')
        
        try:
            # Calculate market volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # STRATEGY 1: DAY TRADING CALLS/PUTS
            # Always include Day Trading Calls/Puts as a high priority strategy
            day_trading_confidence = 0.75  # Base confidence
            
            # Boost confidence based on conditions
            if options_priority:
                day_trading_confidence += 0.15  # Boost if options are prioritized
            
            if trend_strength == 'strong' and volume_trend == 'above_average':
                day_trading_confidence += 0.1  # Boost for strong trend with volume
                
            if volatility > 0.25:
                day_trading_confidence += 0.05  # Boost for higher volatility
            
            strategy_matches.append({
                'name': 'Day Trading Calls/Puts',
                'type': 'long' if trend_direction == 'bullish' else 'short',
                'confidence': min(0.95, day_trading_confidence),  # Cap at 0.95
                'reason': f'Directional options trading ({trend_direction}) with {trend_strength} trend'
            })
            
            # STRATEGY 2: IRON CONDOR
            # Include Iron Condor for neutral/range-bound markets
            iron_condor_confidence = 0.7  # Base confidence
            
            # Adjust confidence based on conditions
            if options_priority:
                iron_condor_confidence += 0.15  # Boost if options are prioritized
                
            if volatility < 0.2 and momentum_condition == 'neutral':
                iron_condor_confidence += 0.1  # Boost for low volatility and neutral momentum
                
            if trend_strength == 'weak':
                iron_condor_confidence += 0.05  # Boost for weak trends (range-bound)
            
            strategy_matches.append({
                'name': 'Iron Condor',
                'type': 'neutral',
                'confidence': min(0.95, iron_condor_confidence),  # Cap at 0.95
                'reason': 'Range-bound market strategy with defined risk/reward'
            })
            
            # STRATEGY 3: SWING TRADING
            # Include Swing Trading for longer-term directional moves
            swing_trading_confidence = 0.65  # Base confidence
            
            # Adjust confidence based on conditions
            if not options_priority:
                swing_trading_confidence += 0.1  # Boost if options are not prioritized
                
            if trend_strength == 'strong' and len(data) > 50:
                swing_trading_confidence += 0.15  # Boost for strong trend on longer timeframe
                
            if volume_trend == 'above_average':
                swing_trading_confidence += 0.05  # Boost for volume confirmation
            
            strategy_matches.append({
                'name': 'Swing Trading',
                'type': 'long' if trend_direction == 'bullish' else 'short',
                'confidence': min(0.95, swing_trading_confidence),  # Cap at 0.95
                'reason': f'Multi-day {trend_direction} trend suitable for swing trading'
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
            
            # Enhance parameters with comprehensive strike selection when applicable
            current_price = self.data['Close'].iloc[-1] if 'Close' in self.data.columns else None
            
            # If we have price data and this is an options strategy, add comprehensive strike selection
            if current_price is not None and strategy_name in [
                'Day Trading Calls/Puts', 'Iron Condor', 'Straddle/Strangle', 
                'Covered Calls', 'Credit Spreads', 'Calendar Spreads'
            ]:
                strike_info = self._get_comprehensive_strike_selection(self.data, strategy_name, current_price)
                parameters['strike_info'] = strike_info['final_recommendation']
            
            # Strategy-specific parameters
            if strategy_name == 'Day Trading Calls/Puts':
                parameters.update({
                    'entry_condition': 'momentum_breakout',
                    'exit_condition': 'same_day_close',
                    'stop_loss': 'tight_atr_based',
                    'profit_target': 'quick_scalp',
                    'max_holding_time': '6_hours',
                    'strike_selection': 'comprehensive'
                })
            
            elif strategy_name == 'Iron Condor':
                parameters.update({
                    'entry_condition': 'high_iv_rank',
                    'exit_condition': 'profit_target_or_time',
                    'strike_selection': 'comprehensive',  # Using the new comprehensive approach
                    'profit_target': '25_percent_max_profit',
                    'time_decay_benefit': True
                })
            
            elif strategy_name == 'Straddle/Strangle':
                parameters.update({
                    'entry_condition': 'volatility_expansion_expected',
                    'exit_condition': 'volatility_crush_or_profit',
                    'strike_selection': 'comprehensive',  # Using the new comprehensive approach
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
                    'strike_selection': 'comprehensive',  # Using the new comprehensive approach
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
            
    def _get_comprehensive_strike_selection(self, data: pd.DataFrame, strategy_type: str, current_price: float) -> dict:
        """
        Comprehensive strike selection method that combines Standard Deviation, 
        Technical Levels, and Delta-Based approaches.
        
        Args:
            data: DataFrame with price and indicator data
            strategy_type: Type of strategy (e.g., 'Iron Condor', 'Covered Call')
            current_price: Current price of the underlying asset
            
        Returns:
            Dictionary with recommended strikes from different methods and a final recommendation
        """
        result = {
            'standard_deviation': {},
            'technical_levels': {},
            'delta_based': {},
            'final_recommendation': {}
        }
        
        try:
            # 1. Standard Deviation based strikes
            if 'HV_20' in data.columns:
                hist_vol = data['HV_20'].iloc[-1] / 100  # Convert from percentage
            else:
                # Estimate historical volatility if not available
                returns = data['Close'].pct_change().dropna()
                hist_vol = returns.std() * (252 ** 0.5)  # Annualized
                
            # Calculate 1 and 2 standard deviation moves (weekly)
            weekly_factor = (7/365)**0.5  # Time scaling
            std_dev_1w = current_price * hist_vol * weekly_factor
            
            result['standard_deviation'] = {
                'half_std': {
                    'up': current_price + (std_dev_1w * 0.5),
                    'down': current_price - (std_dev_1w * 0.5)
                },
                'one_std': {
                    'up': current_price + std_dev_1w,
                    'down': current_price - std_dev_1w
                },
                'two_std': {
                    'up': current_price + (std_dev_1w * 2),
                    'down': current_price - (std_dev_1w * 2)
                }
            }
            
            # 2. Technical Levels based strikes
            result['technical_levels'] = {
                'support': [],
                'resistance': []
            }
            
            # Extract support/resistance levels if available
            if 'levels' in data.attrs and isinstance(data.attrs['levels'], dict):
                result['technical_levels']['support'] = data.attrs['levels'].get('support', [])
                result['technical_levels']['resistance'] = data.attrs['levels'].get('resistance', [])
            
            # 3. Delta-based strikes (estimated)
            # Since we don't have actual options chain data, we'll estimate using volatility
            delta_30_call_strike = current_price * (1 + 0.4 * weekly_factor * hist_vol)
            delta_30_put_strike = current_price * (1 - 0.4 * weekly_factor * hist_vol)
            
            result['delta_based'] = {
                'delta_30': {
                    'call': delta_30_call_strike,
                    'put': delta_30_put_strike
                },
                'delta_16': {
                    'call': current_price * (1 + 0.6 * weekly_factor * hist_vol),
                    'put': current_price * (1 - 0.6 * weekly_factor * hist_vol)
                }
            }
            
            # 4. Make final recommendation based on strategy type
            if strategy_type in ['Iron Condor', 'Short Strangle']:
                # For iron condors, we want balanced strikes that combine all methods
                result['final_recommendation'] = {
                    'call_strike': self._weighted_average([
                        result['standard_deviation']['one_std']['up'],
                        self._closest_level(result['technical_levels']['resistance'], current_price, 'above'),
                        result['delta_based']['delta_30']['call']
                    ]),
                    'put_strike': self._weighted_average([
                        result['standard_deviation']['one_std']['down'],
                        self._closest_level(result['technical_levels']['support'], current_price, 'below'),
                        result['delta_based']['delta_30']['put']
                    ])
                }
                
            elif strategy_type in ['Covered Call', 'Cash-Secured Put']:
                # For income strategies, prefer technical levels with standard deviation validation
                if strategy_type == 'Covered Call':
                    result['final_recommendation']['strike'] = self._weighted_average([
                        result['standard_deviation']['half_std']['up'],
                        self._closest_level(result['technical_levels']['resistance'], current_price, 'above'),
                        result['delta_based']['delta_30']['call']
                    ], weights=[0.3, 0.5, 0.2])
                else:
                    result['final_recommendation']['strike'] = self._weighted_average([
                        result['standard_deviation']['half_std']['down'],
                        self._closest_level(result['technical_levels']['support'], current_price, 'below'),
                        result['delta_based']['delta_30']['put']
                    ], weights=[0.3, 0.5, 0.2])
                    
            elif strategy_type in ['Long Call', 'Long Put']:
                # For directional strategies, blend all methods
                if strategy_type == 'Long Call':
                    result['final_recommendation']['strike'] = self._weighted_average([
                        current_price, # ATM
                        self._closest_level(result['technical_levels']['support'], current_price, 'below'),
                        result['delta_based']['delta_30']['call']
                    ], weights=[0.4, 0.3, 0.3])
                else:
                    result['final_recommendation']['strike'] = self._weighted_average([
                        current_price, # ATM
                        self._closest_level(result['technical_levels']['resistance'], current_price, 'above'),
                        result['delta_based']['delta_30']['put']
                    ], weights=[0.4, 0.3, 0.3])
            else:
                # Default to standard deviation approach
                result['final_recommendation']['upper_strike'] = result['standard_deviation']['one_std']['up']
                result['final_recommendation']['lower_strike'] = result['standard_deviation']['one_std']['down']
                
            # Round strikes to appropriate levels
            for key, value in result['final_recommendation'].items():
                if isinstance(value, (int, float)):
                    # Round to nearest 0.5 or 1.0 depending on price level
                    round_to = 0.5 if current_price < 50 else 1.0
                    result['final_recommendation'][key] = round(value / round_to) * round_to
                    
        except Exception as e:
            print(f"Error in comprehensive strike selection: {e}")
            # Fallback to basic standard deviation method
            result['final_recommendation'] = {
                'upper_strike': round(current_price * 1.05),
                'lower_strike': round(current_price * 0.95)
            }
            
        return result
    
    def _weighted_average(self, values, weights=None):
        """Calculate weighted average of values, ignoring None values."""
        if not values:
            return None
            
        # Filter out None values
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
            
        if weights is None:
            # Equal weights if not specified
            weights = [1/len(valid_values)] * len(valid_values)
            
        # Ensure weights match the number of valid values
        if len(weights) != len(valid_values):
            weights = [1/len(valid_values)] * len(valid_values)
            
        return sum(v * w for v, w in zip(valid_values, weights))
        
    def _closest_level(self, levels, price, direction='above'):
        """Find the closest level above or below the current price."""
        if not levels:
            return None
            
        if direction == 'above':
            above_levels = [level for level in levels if level > price]
            return min(above_levels) if above_levels else None
        else:
            below_levels = [level for level in levels if level < price]
            return max(below_levels) if below_levels else None
