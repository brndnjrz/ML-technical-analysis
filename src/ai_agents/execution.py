import pandas as pd
from typing import Dict, Any, List, Tuple
import numpy as np
from ..trading_strategies import strategies_data

class ExecutionAgent:
    """
    Agent responsible for determining optimal entry/exit points and managing trade execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Load trading strategies for execution-specific rules
        self.strategies_db = {strategy['Strategy']: strategy for strategy in strategies_data}
        
    def generate_signals(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        print("[ExecutionAgent] Generating signals")
        """Generate entry and exit signals based on the strategy."""
        signals = {
            'entry': None,
            'exit': None,
            'position_size': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        try:
            # Generate entry signals
            entry_conditions = self._check_entry_conditions(strategy, data)
            
            # Generate exit signals
            exit_conditions = self._check_exit_conditions(strategy, data)
            
            # Calculate position size
            position_size = self._calculate_position_size(strategy, data)
            
            # Set stop loss and take profit levels
            stop_loss, take_profit = self._calculate_risk_levels(strategy, data)
            
            signals.update({
                'entry': entry_conditions,
                'exit': exit_conditions,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            
        return signals
    
    def _check_entry_conditions(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check if entry conditions are met based on the strategy."""
        entry_signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            strategy_name = strategy.get('name')
            
            # Use trading_strategies.py data for strategy-specific entry conditions
            if strategy_name in self.strategies_db:
                strategy_info = self.strategies_db[strategy_name]
                entry_signal = self._check_options_strategy_entry(strategy_name, strategy_info, data)
            else:
                # Fallback to original logic for non-options strategies
                if strategy_name == 'Trend Following':
                    entry_signal = self._check_trend_following_entry(strategy, data)
                elif strategy_name == 'Mean Reversion':
                    entry_signal = self._check_mean_reversion_entry(strategy, data)
                elif strategy_name == 'Range Trading':
                    entry_signal = self._check_range_trading_entry(strategy, data)
                
        except Exception as e:
            print(f"Error checking entry conditions: {str(e)}")
            
        return entry_signal
    
    def _check_options_strategy_entry(self, strategy_name: str, strategy_info: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check entry conditions for options strategies from trading_strategies.py."""
        entry_signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            current_price = data['Close'].iloc[-1]
            
            if strategy_name == 'Day Trading Calls/Puts':
                # High momentum, volume confirmation needed
                rsi = data.get('RSI', pd.Series([50])).iloc[-1]
                volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
                
                if rsi > 60 and volume_ratio > 1.5:  # Bullish momentum
                    entry_signal.update({
                        'signal': True,
                        'type': 'calls',
                        'price': current_price,
                        'confidence': 0.8,
                        'reasons': ['Strong momentum with RSI > 60', 'Volume 50% above average', 'Intraday breakout pattern']
                    })
                elif rsi < 40 and volume_ratio > 1.5:  # Bearish momentum
                    entry_signal.update({
                        'signal': True,
                        'type': 'puts',
                        'price': current_price,
                        'confidence': 0.8,
                        'reasons': ['Bearish momentum with RSI < 40', 'Volume 50% above average', 'Intraday breakdown pattern']
                    })
            
            elif strategy_name == 'Iron Condor':
                # Low volatility, range-bound market
                atr = data.get('ATR', pd.Series([1])).iloc[-1]
                avg_atr = data.get('ATR', pd.Series([1])).rolling(20).mean().iloc[-1]
                rsi = data.get('RSI', pd.Series([50])).iloc[-1]
                
                if atr < avg_atr * 0.8 and 35 < rsi < 65:  # Low volatility, neutral momentum
                    entry_signal.update({
                        'signal': True,
                        'type': 'neutral',
                        'price': current_price,
                        'confidence': 0.75,
                        'reasons': ['Low volatility environment', 'RSI in neutral range', 'Range-bound price action']
                    })
            
            elif strategy_name == 'Straddle/Strangle':
                # High volatility expected, near support/resistance
                bb_width = self._calculate_bb_width(data)
                upcoming_events = self._check_volatility_events()  # Placeholder for event checking
                
                if bb_width < 0.05 and upcoming_events:  # Low current volatility but events coming
                    entry_signal.update({
                        'signal': True,
                        'type': 'volatility_play',
                        'price': current_price,
                        'confidence': 0.85,
                        'reasons': ['Compressed volatility', 'Upcoming volatility events', 'Earnings or news expected']
                    })
            
            elif strategy_name == 'Swing Trading':
                # Trend confirmation with pullback entry
                ema_20 = data.get('EMA_20', data['Close']).iloc[-1]
                ema_50 = data.get('EMA_50', data['Close']).iloc[-1]
                macd = data.get('MACD', pd.Series([0])).iloc[-1]
                signal_line = data.get('MACD_Signal', pd.Series([0])).iloc[-1]
                
                if current_price > ema_20 > ema_50 and macd > signal_line:  # Bullish setup
                    entry_signal.update({
                        'signal': True,
                        'type': 'long',
                        'price': current_price,
                        'confidence': 0.8,
                        'reasons': ['Price above EMAs', 'EMA alignment bullish', 'MACD bullish crossover']
                    })
                elif current_price < ema_20 < ema_50 and macd < signal_line:  # Bearish setup
                    entry_signal.update({
                        'signal': True,
                        'type': 'short',
                        'price': current_price,
                        'confidence': 0.8,
                        'reasons': ['Price below EMAs', 'EMA alignment bearish', 'MACD bearish crossover']
                    })
            
            elif strategy_name == 'Covered Calls':
                # Bullish but expect sideways/slight up movement
                rsi = data.get('RSI', pd.Series([50])).iloc[-1]
                ema_20 = data.get('EMA_20', data['Close']).iloc[-1]
                
                if current_price > ema_20 and 45 < rsi < 65:  # Mild bullish, not overbought
                    entry_signal.update({
                        'signal': True,
                        'type': 'income',
                        'price': current_price,
                        'confidence': 0.7,
                        'reasons': ['Price above EMA20', 'RSI in neutral zone', 'Good for premium collection']
                    })
            
            elif strategy_name == 'Protective Puts':
                # Market uncertainty, portfolio protection needed
                atr = data.get('ATR', pd.Series([1])).iloc[-1]
                avg_atr = data.get('ATR', pd.Series([1])).rolling(20).mean().iloc[-1]
                
                if atr > avg_atr * 1.3:  # Higher than normal volatility
                    entry_signal.update({
                        'signal': True,
                        'type': 'protection',
                        'price': current_price,
                        'confidence': 0.75,
                        'reasons': ['Elevated volatility', 'Portfolio protection warranted', 'Market uncertainty']
                    })
                    
        except Exception as e:
            print(f"Error checking options strategy entry for {strategy_name}: {str(e)}")
        
        return entry_signal
    
    def _calculate_bb_width(self, data: pd.DataFrame) -> float:
        """Calculate Bollinger Bands width as volatility measure."""
        try:
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns and 'BB_middle' in data.columns:
                bb_upper = data['BB_upper'].iloc[-1]
                bb_lower = data['BB_lower'].iloc[-1]
                bb_middle = data['BB_middle'].iloc[-1]
                
                if bb_middle != 0 and not pd.isna(bb_middle):
                    return (bb_upper - bb_lower) / bb_middle
            return 0.05  # Default medium width
        except:
            return 0.05
    
    def _check_volatility_events(self) -> bool:
        """Check for upcoming volatility events (earnings, etc.)."""
        # Placeholder - in real implementation, this would check earnings calendar, 
        # news events, Fed meetings, etc.
        return False
    
    def _check_exit_conditions(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check if exit conditions are met based on the strategy."""
        exit_signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            if strategy['name'] == 'Trend Following':
                exit_signal = self._check_trend_following_exit(strategy, data)
            elif strategy['name'] == 'Mean Reversion':
                exit_signal = self._check_mean_reversion_exit(strategy, data)
            elif strategy['name'] == 'Range Trading':
                exit_signal = self._check_range_trading_exit(strategy, data)
                
        except Exception as e:
            print(f"Error checking exit conditions: {str(e)}")
            
        return exit_signal
    
    def _calculate_position_size(self, strategy: Dict[str, Any], data: pd.DataFrame) -> float:
        """Calculate the appropriate position size based on risk parameters."""
        position_size = 0.0
        
        try:
            # Get risk management parameters
            max_loss_per_trade = strategy['risk_management']['max_loss_per_trade']
            account_size = self.config.get('account_size', 100000)  # Default 100k
            
            # Calculate position size based on stop loss
            current_price = data['Close'].iloc[-1]
            stop_loss = strategy['risk_management']['stop_loss']
            risk_per_share = abs(current_price - stop_loss)
            
            if risk_per_share > 0:
                max_risk_amount = account_size * max_loss_per_trade
                position_size = max_risk_amount / risk_per_share
                
                # Apply volatility adjustment
                position_size *= strategy['risk_management'].get('position_size', 1.0)
                
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            
        return position_size
    
    def _calculate_risk_levels(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        stop_loss = None
        take_profit = None
        
        try:
            current_price = data['Close'].iloc[-1]
            atr = data['ATR'].iloc[-1]
            
            if strategy['type'] == 'long':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            elif strategy['type'] == 'short':
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
                
        except Exception as e:
            print(f"Error calculating risk levels: {str(e)}")
            
        return stop_loss, take_profit
    
    def _check_trend_following_entry(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check entry conditions for trend following strategy."""
        signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            current_price = data['Close'].iloc[-1]
            ma_20 = data['SMA_20'].iloc[-1]
            ma_50 = data['SMA_50'].iloc[-1]
            
            if strategy['type'] == 'long' and current_price > ma_20 > ma_50:
                signal.update({
                    'signal': True,
                    'type': 'long',
                    'price': current_price,
                    'confidence': 0.8,
                    'reasons': ['Price above both MAs', 'MA20 above MA50']
                })
            elif strategy['type'] == 'short' and current_price < ma_20 < ma_50:
                signal.update({
                    'signal': True,
                    'type': 'short',
                    'price': current_price,
                    'confidence': 0.8,
                    'reasons': ['Price below both MAs', 'MA20 below MA50']
                })
                
        except Exception as e:
            print(f"Error checking trend following entry: {str(e)}")
            
        return signal
    
    def _check_mean_reversion_entry(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check entry conditions for mean reversion strategy."""
        signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        # Implement mean reversion entry logic
        return signal
    
    def _check_range_trading_entry(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check entry conditions for range trading strategy."""
        signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        # Implement range trading entry logic
        return signal
    
    def _check_trend_following_exit(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check exit conditions for trend following strategy."""
        signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        # Implement trend following exit logic
        return signal
    
    def _check_mean_reversion_exit(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check exit conditions for mean reversion strategy."""
        signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        # Implement mean reversion exit logic
        return signal
    
    def _check_range_trading_exit(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Check exit conditions for range trading strategy."""
        signal = {
            'signal': False,
            'type': None,
            'price': None,
            'confidence': 0.0,
            'reasons': []
        }
        # Implement range trading exit logic
        return signal
