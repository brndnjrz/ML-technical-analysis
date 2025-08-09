import pandas as pd
from typing import Dict, Any, List, Tuple
import numpy as np

class ExecutionAgent:
    """
    Agent responsible for determining optimal entry/exit points and managing trade execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
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
            if strategy['name'] == 'Trend Following':
                entry_signal = self._check_trend_following_entry(strategy, data)
            elif strategy['name'] == 'Mean Reversion':
                entry_signal = self._check_mean_reversion_entry(strategy, data)
            elif strategy['name'] == 'Range Trading':
                entry_signal = self._check_range_trading_entry(strategy, data)
                
        except Exception as e:
            print(f"Error checking entry conditions: {str(e)}")
            
        return entry_signal
    
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
