import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Setup logger
logger = logging.getLogger(__name__)

class AnalystAgent:
    """
    Agent responsible for technical and fundamental analysis of market data.
    Provides comprehensive market analysis and identifies patterns/trends.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
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
        """Identify common chart patterns."""
        patterns = []
        try:
            # Example pattern: Double Bottom
            patterns.extend(self._find_double_bottom(data))
            
            # Example pattern: Double Top
            patterns.extend(self._find_double_top(data))
            
        except Exception as e:
            print(f"Error in pattern identification: {str(e)}")
            
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
