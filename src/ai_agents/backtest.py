import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

class BacktestAgent:
    """
    Agent responsible for backtesting strategies and validating their performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.results = {}
        
    def run_backtest(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        print("[BacktestAgent] Skipping backtest (debug mode)")
        return {
            'trades': [],
            'metrics': {},
            'equity_curve': None,
            'validation': {'passed': False, 'reasons': ['Backtest skipped (debug mode)']}
        }
    
    def _generate_signals(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals based on strategy rules."""
        signals = pd.DataFrame(index=data.index)
        
        try:
            if strategy['name'] == 'Trend Following':
                signals = self._generate_trend_following_signals(strategy, data)
            elif strategy['name'] == 'Mean Reversion':
                signals = self._generate_mean_reversion_signals(strategy, data)
            elif strategy['name'] == 'Range Trading':
                signals = self._generate_range_trading_signals(strategy, data)
                
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            
        return signals
    
    def _execute_trades(self, signals: pd.DataFrame, data: pd.DataFrame, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute trades based on signals and strategy parameters."""
        trades = []
        
        try:
            position = 0
            entry_price = 0
            entry_time = None
            
            for idx, row in signals.iterrows():
                # Check for entry signals
                if position == 0 and row.get('entry', 0) != 0:
                    position = row['entry']
                    entry_price = data.loc[idx, 'Close']
                    entry_time = idx
                    
                # Check for exit signals
                elif position != 0 and (row.get('exit', 0) != 0 or self._check_stop_loss(
                    position, entry_price, data.loc[idx, 'Close'], strategy)):
                    
                    exit_price = data.loc[idx, 'Close']
                    pnl = (exit_price - entry_price) * position
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'return': pnl / entry_price
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_time = None
                    
        except Exception as e:
            print(f"Error executing trades: {str(e)}")
            
        return trades
    
    def _calculate_metrics(self, trades: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the strategy."""
        # Initialize default metrics
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_return': 0.0,
            'return_std': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'average_trade_duration': 0.0
        }
        
        try:
            if trades and len(trades) > 0:
                returns = [t['return'] for t in trades]
                pnls = [t['pnl'] for t in trades]
                winning_trades = len([r for r in returns if r > 0])
                
                metrics.update({
                    'total_trades': len(trades),
                    'winning_trades': winning_trades,
                    'losing_trades': len(trades) - winning_trades,
                    'win_rate': winning_trades / len(trades) if len(trades) > 0 else 0.0,
                    'average_return': float(np.mean(returns)) if returns else 0.0,
                    'return_std': float(np.std(returns)) if returns else 0.0,
                    'sharpe_ratio': float(np.mean(returns) / np.std(returns)) if returns and np.std(returns) > 0 else 0.0,
                    'max_drawdown': self._calculate_max_drawdown(pnls) if pnls else 0.0,
                    'profit_factor': (
                        float(sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])))
                        if pnls and sum([p for p in pnls if p < 0]) != 0 else 0.0
                    ),
                    'average_trade_duration': float(np.mean([
                        (t['exit_time'] - t['entry_time']).total_seconds() / 86400  # Convert to days
                        for t in trades
                    ])) if trades else 0.0
                })
                
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            
        return metrics
    
    def _generate_equity_curve(self, trades: List[Dict[str, Any]], data: pd.DataFrame) -> pd.Series:
        """Generate equity curve from trades."""
        equity_curve = pd.Series(index=data.index, data=0.0)
        
        try:
            if trades:
                cumulative_pnl = 0
                for trade in trades:
                    cumulative_pnl += trade['pnl']
                    equity_curve[trade['exit_time']:] = cumulative_pnl
                    
        except Exception as e:
            print(f"Error generating equity curve: {str(e)}")
            
        return equity_curve
    
    def _validate_strategy(self, metrics: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate strategy performance against requirements."""
        validation = {
            'passed': False,
            'reasons': [],
            'metrics_summary': {}
        }
        
        try:
            if not metrics or not isinstance(metrics, dict):
                validation['reasons'].append('No valid metrics available')
                return validation
                
            # Store metrics summary
            validation['metrics_summary'] = {
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0)
            }
            
            # Check minimum requirements
            if metrics.get('total_trades', 0) >= 30:
                validation['reasons'].append('Sufficient number of trades')
            else:
                validation['reasons'].append(f"Insufficient number of trades ({metrics.get('total_trades', 0)})")
                
            if metrics.get('win_rate', 0) >= 0.4:
                validation['reasons'].append('Acceptable win rate')
            else:
                validation['reasons'].append(f"Win rate too low ({metrics.get('win_rate', 0):.2%})")
                
            if metrics.get('sharpe_ratio', 0) >= 1.0:
                validation['reasons'].append('Good risk-adjusted returns')
            else:
                validation['reasons'].append(f"Poor risk-adjusted returns (Sharpe: {metrics.get('sharpe_ratio', 0):.2f})")
                
            if metrics.get('max_drawdown', 1.0) <= 0.2:
                validation['reasons'].append('Acceptable drawdown')
            else:
                validation['reasons'].append(f"Drawdown too high ({metrics.get('max_drawdown', 0):.2%})")
                
            # Set overall validation result
            validation['passed'] = len([r for r in validation['reasons'] if 'insufficient' not in r.lower() and 
                                      'too low' not in r.lower() and 'poor' not in r.lower() and 
                                      'too high' not in r.lower()]) >= 3
                
        except Exception as e:
            print(f"Error validating strategy: {str(e)}")
            
        return validation
    
    def _generate_trend_following_signals(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for trend following strategy."""
        signals = pd.DataFrame(index=data.index)
        # Implement trend following signal generation
        return signals
    
    def _generate_mean_reversion_signals(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for mean reversion strategy."""
        signals = pd.DataFrame(index=data.index)
        # Implement mean reversion signal generation
        return signals
    
    def _generate_range_trading_signals(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for range trading strategy."""
        signals = pd.DataFrame(index=data.index)
        # Implement range trading signal generation
        return signals
    
    def _check_stop_loss(self, position: int, entry_price: float, current_price: float, strategy: Dict[str, Any]) -> bool:
        """Check if stop loss has been hit."""
        if 'risk_management' in strategy and 'stop_loss' in strategy['risk_management']:
            stop_level = strategy['risk_management']['stop_loss']
            if position > 0:  # Long position
                return current_price <= (entry_price - stop_level)
            else:  # Short position
                return current_price >= (entry_price + stop_level)
        return False
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from list of PnLs."""
        cumulative = np.cumsum(pnls)
        max_drawdown = 0
        peak = cumulative[0]
        
        for value in cumulative[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
