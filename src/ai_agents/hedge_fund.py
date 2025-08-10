from typing import Dict, Any, List, Tuple
import pandas as pd
import logging
from .analyst import AnalystAgent
from .strategy import StrategyAgent
from .execution import ExecutionAgent
from .backtest import BacktestAgent

# Setup logger for AI agents
logger = logging.getLogger(__name__)

class HedgeFundAI:
    """
    Main AI class that orchestrates all specialized agents for comprehensive market analysis.
    """
    
    def analyze_and_recommend(self, data: pd.DataFrame, ticker: str, current_price: float) -> Dict[str, Any]:
        logger.debug(f"🤖 HedgeFundAI analyzing {ticker}")
        """
        Main analysis method that provides market analysis and recommendations.
        This is the primary interface used by ai_analysis.py.
        """
        try:
            # Get comprehensive market analysis
            analysis = self.analyze_market(data, ticker)
            
            # Extract relevant information for the recommendation format expected by ai_analysis.py
            market_analysis = {
                'RSI': data['RSI'].iloc[-1] if 'RSI' in data else None,
                'MACD_Signal': 'bullish' if 'MACD' in data and 'MACD_Signal' in data and data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else 'bearish' if 'MACD' in data and 'MACD_Signal' in data else 'neutral',
                'volume_signal': 'high' if 'Volume' in data and data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1] else 'low' if 'Volume' in data else 'unknown',
                'trend_strength': data['ADX'].iloc[-1] if 'ADX' in data else 0.0,
                'trend': analysis.get('market_analysis', {}).get('trend', {}),
                'volatility': analysis.get('market_analysis', {}).get('volatility', {}),
                'momentum': analysis.get('market_analysis', {}).get('momentum', {})
            }
            
            # Format strategy information
            strategy = analysis.get('strategy', {})
            strategy['reason'] = strategy.get('parameters', {}).get('reasons', [])
            if not strategy['reason']:
                strategy['reason'] = [f"Based on {strategy.get('name', 'Unknown')} strategy"]
            
            # Safely get risk_assessment
            risk_assessment = analysis.get('risk_assessment', {'risk_level': 'UNKNOWN', 'confidence': 0.0, 'warnings': [], 'factors': {}})
            
            # Compile the recommendation in the expected format
            recommendation = {
                'market_analysis': market_analysis,
                'strategy': strategy,
                'signals': analysis.get('signals', {}),
                'current_price': current_price,
                'risk_assessment': risk_assessment,
                'action': analysis.get('recommendation', {}).get('action'),
                'confidence': strategy.get('confidence', 0.0),
                'entry_price': analysis.get('recommendation', {}).get('entry_price'),
                'stop_loss': analysis.get('recommendation', {}).get('stop_loss'),
                'take_profit': analysis.get('recommendation', {}).get('take_profit')
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error in analyze_and_recommend: {str(e)}")
            # Return a default structure to prevent crashes
            return {
                'market_analysis': {},
                'strategy': {
                    'name': 'HOLD',
                    'confidence': 0.0,
                    'reason': ['Analysis failed']
                },
                'signals': {},
                'current_price': current_price,
                'risk_assessment': {'risk_level': 'UNKNOWN'},
                'action': 'HOLD',
                'confidence': 0.0
            }
    """
    Main AI class that orchestrates all specialized agents for comprehensive market analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.analyst = AnalystAgent(config)
        self.strategist = StrategyAgent(config)
        self.executor = ExecutionAgent(config)
        self.backtester = BacktestAgent(config)
        
    def analyze_market(self, data: pd.DataFrame, ticker: str, fundamental_data: Dict[str, Any] = None) -> Dict[str, Any]:
        logger.debug(f"📊 Analyzing market conditions for {ticker}")
        """
        Perform comprehensive market analysis using all agents.
        """
        try:
            # Step 1: Market Analysis
            market_analysis = self.analyst.analyze_technical_indicators(data)
            if fundamental_data:
                fundamental_analysis = self.analyst.analyze_fundamental_data(ticker, fundamental_data)
                market_analysis.update({'fundamental': fundamental_analysis})
            
            # Step 2: Strategy Development
            strategy = self.strategist.develop_strategy(market_analysis, data)
            
            # Step 3: Trade Signals
            signals = self.executor.generate_signals(strategy, data)
            
            # Step 4: Strategy Validation
            if len(data) > 30:  # Only backtest if we have enough historical data
                backtest_results = self.backtester.run_backtest(strategy, data)
                strategy['validation'] = backtest_results['validation']
                strategy['metrics'] = backtest_results['metrics']
            
            # Compile final analysis
            analysis = {
                'market_analysis': market_analysis,
                'strategy': strategy,
                'signals': signals,
                'risk_assessment': self._assess_risk(strategy, signals),
                'recommendation': self._generate_recommendation(strategy, signals, market_analysis)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in market analysis: {str(e)}")
            return {}
    
    def _assess_risk(self, strategy: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall risk based on strategy and signals.
        """
        risk_assessment = {
            'risk_level': 'MEDIUM',
            'confidence': 0.0,
            'warnings': [],
            'factors': {}
        }
        
        try:
            # Analyze strategy risk
            strategy_risk = strategy.get('risk_management', {})
            position_size = signals.get('position_size', 0)
            
            # Risk factors
            factors = {
                'position_size': position_size,
                'stop_loss': strategy_risk.get('stop_loss'),
                'max_loss': strategy_risk.get('max_loss_per_trade'),
                'portfolio_risk': strategy_risk.get('max_portfolio_risk')
            }
            
            # Calculate confidence
            confidence_factors = []
            if strategy.get('confidence'):
                confidence_factors.append(strategy['confidence'])
            if signals.get('entry', {}).get('confidence'):
                confidence_factors.append(signals['entry']['confidence'])
            
            risk_assessment.update({
                'risk_level': self._calculate_risk_level(factors),
                'confidence': sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0,
                'factors': factors
            })
            
        except Exception as e:
            print(f"Error in risk assessment: {str(e)}")
            
        return risk_assessment
    
    def _calculate_risk_level(self, factors: Dict[str, Any]) -> str:
        """
        Calculate risk level based on various factors.
        """
        try:
            if factors.get('position_size', 0) > 0.8:
                return 'HIGH'
            elif factors.get('position_size', 0) < 0.3:
                return 'LOW'
            return 'MEDIUM'
        except:
            return 'MEDIUM'
    
    def _generate_recommendation(self, strategy: Dict[str, Any], signals: Dict[str, Any], 
                               market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final trading recommendation.
        """
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasoning': [],
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        try:
            # Check entry signals
            entry = signals.get('entry', {})
            if entry.get('signal'):
                recommendation.update({
                    'action': 'BUY' if entry['type'] == 'long' else 'SELL',
                    'confidence': entry.get('confidence', 0.0),
                    'reasoning': entry.get('reasons', []),
                    'entry_price': entry.get('price'),
                    'stop_loss': signals.get('stop_loss'),
                    'take_profit': signals.get('take_profit')
                })
            else:
                # Check if exit is recommended
                exit_signal = signals.get('exit', {})
                if exit_signal.get('signal'):
                    recommendation.update({
                        'action': 'EXIT',
                        'confidence': exit_signal.get('confidence', 0.0),
                        'reasoning': exit_signal.get('reasons', []),
                        'exit_price': exit_signal.get('price')
                    })
                    
            # Add market context
            if market_analysis:
                recommendation['market_context'] = {
                    'trend': market_analysis.get('trend', {}).get('short_term'),
                    'strength': market_analysis.get('trend', {}).get('strength'),
                    'volatility': market_analysis.get('volatility', {}).get('volatility_state')
                }
                
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            
        return recommendation
