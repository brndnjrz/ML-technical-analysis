from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from src.indicators import calculate_indicators
from src.prediction import create_model, engineer_features

@dataclass
class AnalystAgent:
    """Analyzes technical indicators and market conditions."""
    
    def analyze_market(self, data: pd.DataFrame, timeframe: str = '15m') -> Dict[str, Any]:
        """Analyze current market conditions using technical indicators."""
        analysis = {
            'RSI': data['RSI_14'].iloc[-1],
            'MACD_Signal': 'bullish' if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else 'bearish',
            'ATR_volatility': data['ATR'].iloc[-1],
            'trend_strength': data['ADX'].iloc[-1],
            'volume_signal': 'high' if data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1] else 'low',
            'bb_position': self._analyze_bb_position(data)
        }
        
        return analysis
    
    def _analyze_bb_position(self, data: pd.DataFrame) -> str:
        """Analyze position relative to Bollinger Bands."""
        close = data['Close'].iloc[-1]
        upper = data['BB_upper'].iloc[-1]
        lower = data['BB_lower'].iloc[-1]
        
        if close > upper:
            return 'overbought'
        elif close < lower:
            return 'oversold'
        return 'neutral'

@dataclass
class StrategyAgent:
    """Selects appropriate trading strategies based on market conditions."""
    
    def select_strategy(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most appropriate strategy based on market conditions."""
        
        strategy = {
            'name': None,
            'confidence': 0.0,
            'reason': [],
            'risk_level': 'medium'
        }
        
        # Iron Condor Conditions
        if (20 <= market_analysis['RSI'] <= 80 and 
            market_analysis['trend_strength'] < 25 and 
            market_analysis['bb_position'] == 'neutral'):
            strategy.update({
                'name': 'Iron Condor',
                'confidence': 0.8,
                'reason': ['Range-bound market', 'Moderate volatility', 'No strong trend'],
                'risk_level': 'medium'
            })
            
        # Vertical Spread Conditions
        elif ((market_analysis['RSI'] > 70 or market_analysis['RSI'] < 30) and
              market_analysis['trend_strength'] > 25):
            strategy.update({
                'name': 'Vertical Spread',
                'confidence': 0.85,
                'reason': ['Strong trend', 'Overbought/Oversold conditions'],
                'risk_level': 'medium-high'
            })
            
        # Covered Call Conditions
        elif (market_analysis['trend_strength'] > 20 and
              market_analysis['volume_signal'] == 'high' and
              market_analysis['bb_position'] != 'overbought'):
            strategy.update({
                'name': 'Covered Call',
                'confidence': 0.75,
                'reason': ['Strong uptrend', 'High volume', 'Not overbought'],
                'risk_level': 'low'
            })
            
        return strategy

@dataclass
class ExecutionAgent:
    """Determines specific trade parameters based on selected strategy."""
    
    def calculate_trade_parameters(self, 
                                 data: pd.DataFrame,
                                 strategy: Dict[str, Any],
                                 current_price: float) -> Dict[str, Any]:
        """Calculate specific trade parameters based on strategy."""
        
        atr = data['ATR'].iloc[-1]
        
        parameters = {
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'expiration_days': None,
            'strikes': {}
        }
        
        if strategy['name'] == 'Iron Condor':
            wing_width = atr * 1.5
            parameters.update({
                'strikes': {
                    'call_sell': round(current_price + wing_width, 2),
                    'call_buy': round(current_price + (wing_width * 1.5), 2),
                    'put_sell': round(current_price - wing_width, 2),
                    'put_buy': round(current_price - (wing_width * 1.5), 2)
                },
                'stop_loss': round(current_price * 0.02, 2),  # 2% max loss
                'expiration_days': 7
            })
            
        elif strategy['name'] == 'Vertical Spread':
            direction = 'call' if data['RSI_14'].iloc[-1] < 30 else 'put'
            width = atr * 1.0
            
            if direction == 'call':
                parameters.update({
                    'strikes': {
                        'buy': round(current_price, 2),
                        'sell': round(current_price + width, 2)
                    }
                })
            else:
                parameters.update({
                    'strikes': {
                        'buy': round(current_price, 2),
                        'sell': round(current_price - width, 2)
                    }
                })
            
            parameters.update({
                'stop_loss': round(width * 0.75, 2),
                'expiration_days': 14
            })
            
        return parameters

@dataclass
class BacktestAgent:
    """Backtests strategies on historical data."""
    
    def backtest_strategy(self, 
                         data: pd.DataFrame,
                         strategy: Dict[str, Any],
                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic backtesting of the strategy."""
        
        results = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Implement strategy-specific backtesting logic here
        # This is a placeholder for demonstration
        
        return results

class HedgeFundAI:
    """Main class that coordinates all AI agents."""
    
    def __init__(self):
        self.analyst = AnalystAgent()
        self.strategist = StrategyAgent()
        self.executor = ExecutionAgent()
        self.backtester = BacktestAgent()
    
    def analyze_and_recommend(self, 
                            data: pd.DataFrame,
                            ticker: str,
                            current_price: float) -> Dict[str, Any]:
        """Complete analysis and strategy recommendation."""
        
        # 1. Market Analysis
        market_analysis = self.analyst.analyze_market(data)
        
        # 2. Strategy Selection
        selected_strategy = self.strategist.select_strategy(market_analysis)
        
        # 3. Trade Parameters
        if selected_strategy['name']:
            trade_parameters = self.executor.calculate_trade_parameters(
                data, selected_strategy, current_price
            )
            
            # 4. Backtesting
            backtest_results = self.backtester.backtest_strategy(
                data, selected_strategy, trade_parameters
            )
        else:
            trade_parameters = {}
            backtest_results = {}
        
        # 5. Compile Final Recommendation
        recommendation = {
            'ticker': ticker,
            'timestamp': pd.Timestamp.now(),
            'current_price': current_price,
            'market_analysis': market_analysis,
            'strategy': selected_strategy,
            'parameters': trade_parameters,
            'backtest_results': backtest_results
        }
        
        return recommendation
