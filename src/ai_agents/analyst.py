import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from ..trading_strategies import strategies_data

# Setup logger
logger = logging.getLogger(__name__)

class AnalystAgent:
    """
    Agent responsible for technical and fundamental analysis of market data.
    Provides comprehensive market analysis and identifies patterns/trends.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Load strategies for context-aware analysis
        self.strategies_db = {strategy['Strategy']: strategy for strategy in strategies_data}
    
    def analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.debug("📈 Analyzing technical indicators")
        """Analyze technical indicators and identify significant patterns."""
        analysis = {}
        try:
            # Trend Analysis
            analysis['trend'] = self._analyze_trend(data)
            
            # Momentum Analysis
            analysis['momentum'] = self._analyze_momentum(data)
            
            # Volume Analysis
            analysis['volume'] = self._analyze_volume(data)
            
            # Volatility Analysis
            analysis['volatility'] = self._analyze_volatility(data)
            
            # Pattern Recognition
            analysis['patterns'] = self._identify_patterns(data)
            
        except Exception as e:
            print(f"Error in technical analysis: {str(e)}")
            
        return analysis
    
    def analyze_fundamental_data(self, ticker: str, fundamental_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("📊 Analyzing fundamental data")
        """Analyze fundamental data and provide insights."""
        analysis = {}
        try:
            # Financial Ratios Analysis
            analysis['ratios'] = self._analyze_ratios(fundamental_data)
            
            # Growth Analysis
            analysis['growth'] = self._analyze_growth(fundamental_data)
            
            # Valuation Analysis
            analysis['valuation'] = self._analyze_valuation(fundamental_data)
            
            # Risk Assessment
            analysis['risk'] = self._assess_risk(fundamental_data)
            
        except Exception as e:
            print(f"Error in fundamental analysis: {str(e)}")
            
        return analysis
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends using moving averages and other trend indicators."""
        trend_analysis = {}
        
        try:
            # Short-term trend (20-day MA)
            short_term = data['EMA_20'].iloc[-1] > data['EMA_20'].iloc[-2]
            
            # Medium-term trend (50-day MA)
            medium_term = data['EMA_50'].iloc[-1] > data['EMA_50'].iloc[-2]
            
            # Trend strength (ADX)
            trend_strength = data['ADX'].iloc[-1]
            
            trend_analysis = {
                'short_term': 'bullish' if short_term else 'bearish',
                'medium_term': 'bullish' if medium_term else 'bearish',
                'strength': trend_strength,
                'strength_level': 'strong' if trend_strength > 25 else 'weak'
            }
        except Exception as e:
            print(f"Error in trend analysis: {str(e)}")
            
        return trend_analysis
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price momentum using RSI, MACD, and other momentum indicators."""
        momentum_analysis = {}
        
        try:
            # RSI Analysis
            rsi = data['RSI'].iloc[-1]
            
            # MACD Analysis
            macd = data['MACD'].iloc[-1]
            signal = data['MACD_Signal'].iloc[-1]
            
            momentum_analysis = {
                'rsi': {
                    'value': rsi,
                    'condition': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
                },
                'macd': {
                    'value': macd,
                    'signal': signal,
                    'crossover': 'bullish' if macd > signal else 'bearish'
                }
            }
        except Exception as e:
            print(f"Error in momentum analysis: {str(e)}")
            
        return momentum_analysis
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trends and patterns."""
        volume_analysis = {}
        
        try:
            # Current volume vs Average
            curr_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            
            # OBV Trend
            obv_trend = data['OBV'].iloc[-1] > data['OBV'].iloc[-2]
            
            volume_analysis = {
                'volume_ratio': curr_volume / avg_volume,
                'volume_trend': 'above_average' if curr_volume > avg_volume else 'below_average',
                'obv_trend': 'increasing' if obv_trend else 'decreasing'
            }
        except Exception as e:
            print(f"Error in volume analysis: {str(e)}")
            
        return volume_analysis
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility using ATR and Bollinger Bands."""
        volatility_analysis = {}
        
        try:
            # ATR Analysis
            atr = data['ATR'].iloc[-1]
            avg_atr = data['ATR'].rolling(window=20).mean().iloc[-1]
            
            # Bollinger Bands Analysis
            bb_middle = data['BB_middle'].iloc[-1]
            bb_upper = data['BB_upper'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            # Avoid division by zero or NaN
            if bb_middle is not None and bb_middle != 0 and not pd.isna(bb_middle):
                bb_width = (bb_upper - bb_lower) / bb_middle
            else:
                bb_width = float('nan')
            
            volatility_analysis = {
                'atr': {
                    'value': atr,
                    'relative_to_average': 'high' if atr > avg_atr else 'low'
                },
                'bb_width': bb_width,
                'volatility_state': 'high' if bb_width > 0.05 else 'low'
            }
        except Exception as e:
            print(f"Error in volatility analysis: {str(e)}")
            
        return volatility_analysis
    
    def _identify_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify common chart patterns relevant to trading strategies."""
        patterns = []
        try:
            # Strategy-aware pattern recognition
            patterns.extend(self._find_volatility_patterns(data))  # For Straddle/Strangle strategies
            patterns.extend(self._find_trend_patterns(data))       # For Swing Trading, Day Trading
            patterns.extend(self._find_range_patterns(data))       # For Iron Condor, Butterfly
            patterns.extend(self._find_breakout_patterns(data))    # For momentum strategies
            
            # Original patterns
            patterns.extend(self._find_double_bottom(data))
            patterns.extend(self._find_double_top(data))
            
        except Exception as e:
            print(f"Error in pattern identification: {str(e)}")
            
        return patterns
    
    def _find_volatility_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify volatility patterns relevant for Straddle/Strangle strategies."""
        patterns = []
        try:
            # Calculate volatility compression/expansion
            if 'ATR' in data.columns:
                atr = data['ATR']
                atr_ma = atr.rolling(20).mean()
                current_atr = atr.iloc[-1]
                avg_atr = atr_ma.iloc[-1]
                
                if current_atr < avg_atr * 0.7:  # Volatility compression
                    patterns.append({
                        'pattern': 'Volatility Compression',
                        'strategy_relevance': ['Straddle/Strangle', 'Calendar Spreads'],
                        'description': 'Low volatility may precede expansion',
                        'confidence': 0.7,
                        'time_horizon': 'short_term'
                    })
                elif current_atr > avg_atr * 1.3:  # Volatility expansion
                    patterns.append({
                        'pattern': 'Volatility Expansion',
                        'strategy_relevance': ['Protective Puts', 'Day Trading Calls/Puts'],
                        'description': 'High volatility environment detected',
                        'confidence': 0.8,
                        'time_horizon': 'immediate'
                    })
        except Exception as e:
            print(f"Error finding volatility patterns: {str(e)}")
        return patterns
    
    def _find_trend_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify trend patterns relevant for trend-following strategies."""
        patterns = []
        try:
            # Moving average alignment
            if all(col in data.columns for col in ['EMA_20', 'EMA_50']):
                ema_20 = data['EMA_20'].iloc[-1]
                ema_50 = data['EMA_50'].iloc[-1]
                current_price = data['Close'].iloc[-1]
                
                if current_price > ema_20 > ema_50:  # Bullish alignment
                    patterns.append({
                        'pattern': 'Bullish Trend Alignment',
                        'strategy_relevance': ['Swing Trading', 'Covered Calls', 'Day Trading Calls/Puts'],
                        'description': 'Price above both EMAs with bullish alignment',
                        'confidence': 0.8,
                        'time_horizon': 'medium_term'
                    })
                elif current_price < ema_20 < ema_50:  # Bearish alignment
                    patterns.append({
                        'pattern': 'Bearish Trend Alignment',
                        'strategy_relevance': ['Swing Trading', 'Protective Puts', 'Day Trading Calls/Puts'],
                        'description': 'Price below both EMAs with bearish alignment',
                        'confidence': 0.8,
                        'time_horizon': 'medium_term'
                    })
        except Exception as e:
            print(f"Error finding trend patterns: {str(e)}")
        return patterns
    
    def _find_range_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify range-bound patterns relevant for neutral strategies."""
        patterns = []
        try:
            # Check for sideways movement
            if 'High' in data.columns and 'Low' in data.columns:
                recent_highs = data['High'].rolling(20).max()
                recent_lows = data['Low'].rolling(20).min()
                current_range = recent_highs.iloc[-1] - recent_lows.iloc[-1]
                avg_range = (recent_highs.rolling(5).std() + recent_lows.rolling(5).std()).iloc[-1]
                
                if avg_range < current_range * 0.3:  # Low range variance = consolidation
                    patterns.append({
                        'pattern': 'Range-Bound Consolidation',
                        'strategy_relevance': ['Iron Condor', 'Butterfly Spread', 'Calendar Spreads'],
                        'description': 'Price consolidating in tight range',
                        'confidence': 0.75,
                        'time_horizon': 'short_to_medium_term'
                    })
        except Exception as e:
            print(f"Error finding range patterns: {str(e)}")
        return patterns
    
    def _find_breakout_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential breakout patterns for momentum strategies."""
        patterns = []
        try:
            # Volume and price breakout detection
            if 'Volume' in data.columns and 'High' in data.columns:
                recent_high = data['High'].rolling(20).max().iloc[-2]  # Previous 20-day high
                current_price = data['Close'].iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                
                if current_price > recent_high and current_volume > avg_volume * 1.5:
                    patterns.append({
                        'pattern': 'Bullish Breakout with Volume',
                        'strategy_relevance': ['Day Trading Calls/Puts', 'Swing Trading'],
                        'description': 'Price breaking above resistance with high volume',
                        'confidence': 0.85,
                        'time_horizon': 'immediate_to_short_term'
                    })
        except Exception as e:
            print(f"Error finding breakout patterns: {str(e)}")
        return patterns
    
    def _find_double_bottom(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential double bottom patterns."""
        patterns = []
        # Implementation for double bottom pattern recognition
        return patterns
    
    def _find_double_top(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potential double top patterns."""
        patterns = []
        # Implementation for double top pattern recognition
        return patterns
    
    def _analyze_ratios(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial ratios."""
        return {}  # Implement ratio analysis
    
    def _analyze_growth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics."""
        return {}  # Implement growth analysis
    
    def _analyze_valuation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics."""
        return {}  # Implement valuation analysis
    
    def _assess_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess fundamental risk factors."""
        return {}  # Implement risk assessment
