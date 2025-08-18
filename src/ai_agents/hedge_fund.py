from typing import Dict, Any, List, Tuple
import pandas as pd
import logging
import numpy as np
from .analyst import AnalystAgent
from .strategy import StrategyAgent
from .execution import ExecutionAgent
from .backtest import BacktestAgent
from ..trading_strategies import strategies_data
from ..utils.metrics import log_prediction

# Setup logger for AI agents
logger = logging.getLogger(__name__)

class HedgeFundAI:
    """
    HEDGE FUND AI ORCHESTRATOR
    
    This class coordinates multiple AI agents to work together like a professional hedge fund:
    - AnalystAgent: Market research and technical analysis
    - StrategyAgent: Strategy selection and optimization  
    - ExecutionAgent: Entry/exit timing and risk management
    - BacktestAgent: Performance validation
    
    The system uses ensemble decision-making and consensus-building to avoid conflicting recommendations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.analyst = AnalystAgent(config)
        self.strategist = StrategyAgent(config)
        self.executor = ExecutionAgent(config)
        self.backtester = BacktestAgent(config)
        # Load strategies database for enhanced recommendations
        self.strategies_db = {strategy['Strategy']: strategy for strategy in strategies_data}
        
        # Hedge Fund Configuration
        self.consensus_threshold = 0.6  # Minimum agreement level for recommendations
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% max position size
            'max_portfolio_vol': 0.15,  # 15% max portfolio volatility
            'min_confidence': 0.5       # 50% minimum confidence for trades
        }
    
    def detect_regime(self, data: pd.DataFrame, ticker: str = "", options_data: Dict[str, Any] = None) -> str:
        """
        Detect current market regime for regime-aware analysis
        
        Returns:
            str: 'trend', 'range', or 'event'
        """
        try:
            if len(data) < 50:
                logger.warning("⚠️ Insufficient data for regime detection, defaulting to 'range'")
                return 'range'
            
            # Get technical indicators
            current_price = data['Close'].iloc[-1]
            
            # ADX for trend strength (try multiple column names)
            adx = 20.0  # Default
            for adx_col in ['ADX', 'ADX_14', 'ADX_1d']:
                if adx_col in data.columns:
                    adx = data[adx_col].iloc[-1]
                    break
            
            # Moving average slope for trend direction
            if 'SMA_20' in data.columns or 'EMA_20' in data.columns:
                ma_col = 'SMA_20' if 'SMA_20' in data.columns else 'EMA_20'
                ma_current = data[ma_col].iloc[-1]
                ma_prev = data[ma_col].iloc[-6] if len(data) > 6 else ma_current  # 5-day slope
                ma_slope = (ma_current - ma_prev) / ma_prev if ma_prev != 0 else 0
                slope_threshold = 0.02  # 2% change over 5 days
            else:
                # Fallback: price slope
                price_prev = data['Close'].iloc[-6] if len(data) > 6 else current_price
                ma_slope = (current_price - price_prev) / price_prev if price_prev != 0 else 0
                slope_threshold = 0.03  # 3% price change
            
            # Bollinger Band position for range detection
            bb_position = 0.5  # Default neutral
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                bb_upper = data['BB_upper'].iloc[-1]
                bb_lower = data['BB_lower'].iloc[-1]
                if bb_upper != bb_lower:
                    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            # Realized volatility vs Implied volatility for event detection
            realized_vol = data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(data) > 20 else 0.2
            implied_vol = 0.2  # Default
            
            if options_data and 'iv_data' in options_data:
                iv_data = options_data['iv_data']
                if 'iv_percentile' in iv_data:
                    implied_vol = iv_data['iv_percentile'] / 100
                elif 'hv_30' in iv_data:
                    implied_vol = iv_data['hv_30'] / 100
            
            vol_ratio = implied_vol / realized_vol if realized_vol > 0 else 1.0
            
            # Check for upcoming events (earnings, etc.)
            # This is a placeholder - in production, you'd check an events calendar
            days_to_earnings = self._estimate_days_to_earnings(ticker)  # Simplified
            
            # REGIME DECISION LOGIC
            
            # Event regime: High IV vs RV or near earnings
            if vol_ratio > 1.4 or days_to_earnings <= 7:
                regime = 'event'
                logger.info(f"📅 REGIME: Event ({ticker}) - IV/RV: {vol_ratio:.2f}, Days to earnings: ~{days_to_earnings}")
            
            # Trend regime: Strong ADX + significant slope
            elif adx >= 20 and abs(ma_slope) > slope_threshold:
                regime = 'trend'
                logger.info(f"📈 REGIME: Trend ({ticker}) - ADX: {adx:.1f}, Slope: {ma_slope*100:.1f}%")
            
            # Range regime: Low ADX + oscillatory price action
            else:
                regime = 'range'
                logger.info(f"📊 REGIME: Range ({ticker}) - ADX: {adx:.1f}, BB Position: {bb_position:.2f}")
            
            return regime
            
        except Exception as e:
            logger.error(f"❌ Regime detection error: {e}")
            return 'range'  # Safe default
    
    def _estimate_days_to_earnings(self, ticker: str) -> int:
        """
        Estimate days until next earnings announcement
        This is a simplified placeholder - in production, use actual earnings calendar
        """
        # Most companies report quarterly, roughly every 90 days
        # This is a simplified heuristic
        import hashlib
        hash_val = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
        return (hash_val % 90) + 1  # Random but deterministic per ticker
    
    def fuse_agent_probabilities(self, analyst_view: Dict[str, Any], 
                                vision_analysis: Dict[str, Any] = None, 
                                regime: str = 'range') -> Dict[str, float]:
        """
        Fuse quantitative and vision analysis using regime-aware weights
        
        Returns calibrated probabilities for directional and range predictions
        """
        try:
            # Extract quantitative signals
            quant_up_prob = self._extract_quant_up_probability(analyst_view)
            quant_inside_prob = self._extract_quant_inside_probability(analyst_view)
            
            # Extract vision signals (if available)
            vision_up_prob = 0.5
            vision_inside_prob = 0.5
            vision_confidence = 0.0
            
            if vision_analysis:
                vision_up_prob = self._extract_vision_up_probability(vision_analysis)
                vision_inside_prob = self._extract_vision_inside_probability(vision_analysis)
                vision_confidence = vision_analysis.get('confidence', 0.0)
            
            # Regime-specific weights
            regime_weights = {
                'trend': {'quant': 0.7, 'vision': 0.3},
                'range': {'quant': 0.45, 'vision': 0.55},
                'event': {'quant': 0.6, 'vision': 0.4}
            }
            
            weights = regime_weights.get(regime, regime_weights['range'])
            
            # Apply vision confidence scaling
            effective_vision_weight = weights['vision'] * max(0.1, vision_confidence)
            effective_quant_weight = 1.0 - effective_vision_weight
            
            # Fused probabilities
            fused_up_prob = (
                effective_quant_weight * quant_up_prob + 
                effective_vision_weight * vision_up_prob
            )
            
            fused_inside_prob = (
                effective_quant_weight * quant_inside_prob + 
                effective_vision_weight * vision_inside_prob
            )
            
            # Apply regime adjustments
            if regime == 'trend':
                # In trending markets, reduce inside probability
                fused_inside_prob *= 0.8
            elif regime == 'range':
                # In ranging markets, increase inside probability
                fused_inside_prob = min(0.9, fused_inside_prob * 1.2)
            elif regime == 'event':
                # In event-driven markets, increase uncertainty
                fused_up_prob = 0.3 * fused_up_prob + 0.7 * 0.5  # Pull toward neutral
            
            logger.info(f"🔗 FUSION ({regime}): Up={fused_up_prob:.3f}, Inside={fused_inside_prob:.3f} "
                       f"[Weights: Q={effective_quant_weight:.2f}, V={effective_vision_weight:.2f}]")
            
            return {
                'up_probability': np.clip(fused_up_prob, 0.1, 0.9),
                'inside_probability': np.clip(fused_inside_prob, 0.1, 0.9),
                'regime': regime,
                'quant_weight': effective_quant_weight,
                'vision_weight': effective_vision_weight,
                'vision_confidence': vision_confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Fusion error: {e}")
            return {
                'up_probability': 0.5,
                'inside_probability': 0.5,
                'regime': regime,
                'quant_weight': 1.0,
                'vision_weight': 0.0,
                'vision_confidence': 0.0
            }
    
    def _extract_quant_up_probability(self, analyst_view: Dict[str, Any]) -> float:
        """Convert quantitative signals to up probability"""
        try:
            # Get trend and momentum indicators
            trend_data = analyst_view.get('trend', {})
            momentum_data = analyst_view.get('momentum', {})
            
            # Start with neutral
            prob = 0.5
            
            # RSI contribution
            rsi = momentum_data.get('RSI', 50)
            if rsi < 30:
                prob += 0.15  # Oversold bounce
            elif rsi > 70:
                prob -= 0.15  # Overbought pullback
            
            # Trend direction
            trend_direction = trend_data.get('direction', 'neutral')
            if trend_direction == 'bullish':
                prob += 0.2
            elif trend_direction == 'bearish':
                prob -= 0.2
            
            # MACD signal
            momentum_direction = momentum_data.get('direction', 'neutral')
            if momentum_direction == 'bullish':
                prob += 0.1
            elif momentum_direction == 'bearish':
                prob -= 0.1
            
            return np.clip(prob, 0.1, 0.9)
            
        except Exception as e:
            logger.error(f"❌ Quant probability extraction error: {e}")
            return 0.5
    
    def _extract_quant_inside_probability(self, analyst_view: Dict[str, Any]) -> float:
        """Estimate range-bound probability from technical indicators"""
        try:
            # Default range probability
            prob = 0.5
            
            # Volatility indicators suggest range-bound behavior
            volatility_data = analyst_view.get('volatility', {})
            
            # If we have ADX, use it for trend strength assessment
            trend_data = analyst_view.get('trend', {})
            trend_strength = trend_data.get('strength', 0.5)
            
            # Lower trend strength = higher inside probability
            prob = 0.8 - (trend_strength * 0.3)  # Strong trend reduces range probability
            
            # RSI in middle range suggests consolidation
            momentum_data = analyst_view.get('momentum', {})
            rsi = momentum_data.get('RSI', 50)
            
            if 35 <= rsi <= 65:
                prob += 0.1  # RSI in neutral zone
            
            return np.clip(prob, 0.2, 0.9)
            
        except Exception as e:
            logger.error(f"❌ Quant inside probability extraction error: {e}")
            return 0.5
    
    def _extract_vision_up_probability(self, vision_analysis: Dict[str, Any]) -> float:
        """Convert vision analysis to up probability"""
        try:
            trend = vision_analysis.get('trend', 'neutral')
            confidence = vision_analysis.get('confidence', 0.5)
            
            if trend == 'bullish':
                return 0.5 + (confidence * 0.4)  # 0.5 to 0.9
            elif trend == 'bearish':
                return 0.5 - (confidence * 0.4)  # 0.1 to 0.5
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _extract_vision_inside_probability(self, vision_analysis: Dict[str, Any]) -> float:
        """Estimate range probability from vision analysis"""
        try:
            # If we have clear support/resistance levels, higher inside probability
            support_levels = vision_analysis.get('support', [])
            resistance_levels = vision_analysis.get('resistance', [])
            confidence = vision_analysis.get('confidence', 0.5)
            
            if support_levels and resistance_levels:
                # Strong levels = higher inside probability
                return 0.5 + (confidence * 0.3)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def analyze_and_recommend(self, data: pd.DataFrame, ticker: str, current_price: float, options_priority: bool = False, options_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        MAIN HEDGE FUND ANALYSIS ENGINE
        
        Coordinates all agents to provide unified, consensus-driven recommendations.
        Uses ensemble methodology to avoid conflicting signals.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price and indicator data
        ticker : str
            Stock symbol
        current_price : float
            Current market price
        options_priority : bool
            Whether to prioritize options strategies
        options_data : Dict[str, Any], optional
            Options chain data if available (implied volatility, chains, etc.)
        """
        logger.info(f"🏛️ HEDGE FUND AI: Analyzing {ticker} at ${current_price:.2f}")
        
        try:
            # PHASE 0: Market Regime Detection
            logger.debug("🔍 Phase 0: Detecting market regime...")
            regime = self.detect_regime(data, ticker, options_data)
            
            # PHASE 1: Independent Agent Analysis
            logger.debug("📊 Phase 1: Gathering independent agent analyses...")
            
            # Get analysis from each specialist agent
            analyst_view = self.analyst.analyze_technical_indicators(data)
            
                        # Add options priority to strategy agent if requested
            if options_priority:
                self.config['prioritize_options_strategies'] = True
                
                # Analyze options indicators to determine best strategy focus
                if 'IC_SUITABILITY' in data.columns:
                    ic_score = data['IC_SUITABILITY'].iloc[-1]
                    
                    # High IC suitability score (>70) suggests iron condor strategies
                    if ic_score > 70:
                        self.config['preferred_strategies'] = ['Iron Condors', 'Credit Spreads']
                        logger.info(f"📈 Options strategies prioritized: Iron Condors (Suitability: {ic_score:.1f})")
                    # Medium IC suitability (40-70) suggests covered call strategies
                    elif ic_score > 40:
                        self.config['preferred_strategies'] = ['Covered Calls', 'Cash-Secured Puts', 'Credit Spreads']
                        logger.info(f"📈 Options strategies prioritized: Income Strategies (Suitability: {ic_score:.1f})")
                    # Low IC suitability (<40) suggests directional strategies
                    else:
                        self.config['preferred_strategies'] = ['Day Trading Calls/Puts', 'Swing Trading']
                        logger.info(f"📈 Options strategies prioritized: Directional Calls/Puts (Suitability: {ic_score:.1f})")
                else:
                    # Default if no specific indicators available
                    self.config['preferred_strategies'] = ['Day Trading Calls/Puts', 'Iron Condors', 'Covered Calls']
                    logger.info("📈 Options strategies prioritized: Mixed options strategies")
            else:
                # Still use the selected strategies but without preference
                self.config['prioritize_options_strategies'] = False
                self.config['preferred_strategies'] = ['Swing Trading', 'Day Trading Calls/Puts']
                logger.info("📈 Using balanced strategy selection (no options priority)")
                
            strategy_view = self.strategist.develop_strategy(analyst_view, data, options_priority, options_data)
            execution_view = self.executor.generate_signals(strategy_view, data)
            
            # PHASE 2: Consensus Building
            logger.debug("🤝 Phase 2: Building consensus across agents...")
            consensus_result = self._build_consensus({
                'analyst': analyst_view,
                'strategist': strategy_view, 
                'executor': execution_view
            }, data, ticker, regime)  # Pass regime to consensus building
            
            # PHASE 3: Risk Assessment & Final Recommendation
            logger.debug("⚖️ Phase 3: Final risk assessment and recommendation...")
            final_recommendation = self._generate_hedge_fund_recommendation(
                consensus_result, data, ticker, current_price, regime
            )
            
            # PHASE 4: Log prediction for accuracy tracking
            logger.debug("📊 Phase 4: Logging prediction for accuracy measurement...")
            market_data = {
                'current_price': current_price,
                'RSI': data['RSI'].iloc[-1] if 'RSI' in data.columns and len(data) > 0 else 50,
                'ATR': data['ATR'].iloc[-1] if 'ATR' in data.columns and len(data) > 0 else current_price * 0.02,
                'iv_rank': options_data.get('iv_data', {}).get('iv_rank', 0) if options_data else 0,
                'ADX': data['ADX'].iloc[-1] if 'ADX' in data.columns and len(data) > 0 else 25
            }
            
            prediction_id = log_prediction(
                ticker=ticker,
                recommendation=final_recommendation,
                regime=regime,
                market_data=market_data,
                vision_enabled=hasattr(final_recommendation, 'vision_analysis'),
                prompt_version="vA"  # Will be configurable later
            )
            
            final_recommendation['prediction_id'] = prediction_id
            final_recommendation['regime'] = regime
            
            logger.info(f"✅ CONSENSUS REACHED: {final_recommendation['action']} {ticker} "
                       f"(Confidence: {final_recommendation['strategy']['confidence']*100:.0f}%, Regime: {regime})")
            
            return final_recommendation
            
        except Exception as e:
            logger.error(f"❌ Hedge Fund AI Error: {str(e)}")
            return self._generate_fallback_recommendation(data, ticker, current_price)
    
    def _build_consensus(self, agent_views: Dict[str, Any], data: pd.DataFrame, ticker: str, regime: str = 'range') -> Dict[str, Any]:
        """
        BUILD CONSENSUS ACROSS ALL AGENTS
        
        Like a real hedge fund investment committee, this evaluates all viewpoints
        and builds a unified strategy that all agents can support.
        """
        logger.debug("🏛️ Investment Committee: Building consensus...")
        
        consensus = {
            'market_conditions': self._consensus_market_analysis(agent_views, data),
            'strategy_selection': self._consensus_strategy_selection(agent_views, data),
            'risk_assessment': self._consensus_risk_assessment(agent_views, data),
            'execution_plan': self._consensus_execution_plan(agent_views, data),
            'agreement_score': 0.0,
            'conflicting_views': [],
            'final_decision': None
        }
        
        try:
            # Calculate agreement score across all agents
            agreement_scores = []
            
            # Market direction agreement
            market_signals = [
                agent_views.get('analyst', {}).get('trend', {}).get('direction'),
                agent_views.get('strategist', {}).get('type'),
                agent_views.get('executor', {}).get('entry', {}).get('type')
            ]
            direction_agreement = self._calculate_signal_agreement(market_signals)
            agreement_scores.append(direction_agreement)
            
            # Strategy type agreement  
            strategy_preferences = [
                agent_views.get('strategist', {}).get('name'),
                agent_views.get('analyst', {}).get('recommended_strategy'),
                agent_views.get('executor', {}).get('preferred_strategy')
            ]
            strategy_agreement = self._calculate_strategy_agreement(strategy_preferences)
            agreement_scores.append(strategy_agreement)
            
            # Risk level agreement
            risk_assessments = [
                agent_views.get('analyst', {}).get('risk_level'),
                agent_views.get('strategist', {}).get('risk_level'), 
                agent_views.get('executor', {}).get('risk_level')
            ]
            risk_agreement = self._calculate_risk_agreement(risk_assessments)
            agreement_scores.append(risk_agreement)
            
            # Overall consensus score
            consensus['agreement_score'] = np.mean([score for score in agreement_scores if score is not None])
            
            # Identify conflicts
            if consensus['agreement_score'] < self.consensus_threshold:
                consensus['conflicting_views'] = self._identify_conflicts(agent_views)
                logger.warning(f"⚠️ Low consensus ({consensus['agreement_score']:.2f}) - resolving conflicts...")
                consensus = self._resolve_conflicts(consensus, agent_views, data)
            
            consensus['final_decision'] = self._make_final_decision(consensus, agent_views)
            
            # Add consensus metadata for reporting
            consensus['consensus_reached'] = consensus['agreement_score'] >= self.consensus_threshold
            consensus['threshold'] = self.consensus_threshold
            consensus['conflicts'] = consensus.get('conflicting_views', [])
            
        except Exception as e:
            logger.error(f"❌ Consensus building error: {str(e)}")
            consensus['agreement_score'] = 0.5  # Default moderate agreement
            
        return consensus
    
    def _consensus_market_analysis(self, agent_views: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Synthesize market analysis from all agents"""
        try:
            analyst_data = agent_views.get('analyst', {})
            
            # Extract current market metrics
            current_price = data['Close'].iloc[-1] if len(data) > 0 else 0.0
            logger.info(f"🏛️ HEDGE FUND AI: Analyzing {data.name if hasattr(data, 'name') else 'ticker'} at ${current_price:.2f}")
            
            # RSI - Try multiple common column names
            rsi = 50.0  # Default neutral RSI
            for rsi_col in ['RSI', 'RSI_14', 'RSI_1d', 'RSI (14)']:
                if rsi_col in data.columns and len(data) > 0:
                    rsi = data[rsi_col].iloc[-1]
                    break
            
            # ATR - Find ATR for volatility calculation
            atr = 0.0
            for atr_col in ['ATR', 'ATR_14', 'ATR_1d']:
                if atr_col in data.columns and len(data) > 0:
                    atr = data[atr_col].iloc[-1]
                    break
            
            # MACD Signal Analysis
            macd_signal = 'neutral'
            for macd_col in ['MACD', 'MACD_1d']:
                if macd_col in data.columns and len(data) > 0:
                    signal_col = f"{macd_col.split('_')[0]}_Signal" if '_' in macd_col else 'MACD_Signal'
                    if signal_col in data.columns:
                        macd_val = data[macd_col].iloc[-1]
                        macd_sig = data[signal_col].iloc[-1]
                        macd_signal = 'bullish' if macd_val > macd_sig else 'bearish'
                        break
            
            # Volume Analysis
            volume_signal = 'unknown'
            if 'Volume' in data and len(data) > 20:
                current_vol = data['Volume'].iloc[-1]
                avg_vol = data['Volume'].rolling(20).mean().iloc[-1]
                volume_signal = 'high' if current_vol > avg_vol * 1.2 else 'low' if current_vol < avg_vol * 0.8 else 'normal'
            
            # Trend Strength - Try multiple ADX column names
            trend_strength = 0.0
            for adx_col in ['ADX', 'ADX_14', 'ADX_1d']:
                if adx_col in data and len(data) > 0:
                    trend_strength = data[adx_col].iloc[-1]
                    break
            
            return {
                'RSI': rsi,
                'MACD_Signal': macd_signal,
                'volume_signal': volume_signal,
                'trend_strength': trend_strength,
                'trend': analyst_data.get('trend', {}),
                'volatility': analyst_data.get('volatility', {}),
                'momentum': analyst_data.get('momentum', {}),
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Market analysis consensus error: {str(e)}")
            return {'RSI': 50.0, 'MACD_Signal': 'neutral', 'volume_signal': 'unknown', 'trend_strength': 0.0}
    
    def _consensus_strategy_selection(self, agent_views: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Select optimal strategy based on agent consensus"""
        try:
            strategist_rec = agent_views.get('strategist', {})
            
            # Primary strategy from strategist
            primary_strategy = strategist_rec.get('name', 'Day Trading Calls/Puts')
            confidence = strategist_rec.get('confidence', 0.5)
            
            # Validate with analyst and executor
            if primary_strategy in self.strategies_db:
                strategy_info = self.strategies_db[primary_strategy]
                
                # Enhance with detailed strategy information
                enhanced_strategy = {
                    'name': primary_strategy,
                    'confidence': confidence,
                    'type': strategist_rec.get('type', 'momentum'),
                    'description': strategy_info['Description'],
                    'timeframe': strategy_info['Timeframe'],
                    'pros': strategy_info['Pros'],
                    'cons': strategy_info['Cons'],
                    'parameters': self._enhance_strategy_parameters(strategist_rec, strategy_info),
                    'risk_profile': self._assess_strategy_risk(strategy_info, data)
                }
                
                return enhanced_strategy
            else:
                # Fallback to basic strategy
                return {
                    'name': primary_strategy,
                    'confidence': confidence,
                    'type': 'momentum',
                    'parameters': strategist_rec.get('parameters', {}),
                    'risk_profile': 'MEDIUM'
                }
                
        except Exception as e:
            logger.error(f"❌ Strategy selection error: {str(e)}")
            return {
                'name': 'Day Trading Calls/Puts',
                'confidence': 0.5,
                'type': 'momentum',
                'parameters': {},
                'risk_profile': 'MEDIUM'
            }
    
    def _consensus_risk_assessment(self, agent_views: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall risk based on all agent inputs"""
        try:
            # Collect risk factors from all agents - use flexible column names
            volatility = 0.02  # Default
            for atr_col in ['ATR', 'ATR_14', 'ATR_1d']:
                if atr_col in data and len(data) > 0:
                    volatility = data[atr_col].iloc[-1] / data['Close'].iloc[-1]
                    break
            
            # Get trend strength with flexible column names
            trend_strength = 0.0
            for adx_col in ['ADX', 'ADX_14', 'ADX_1d']:
                if adx_col in data and len(data) > 0:
                    trend_strength = data[adx_col].iloc[-1]
                    break
            
            # Calculate composite risk score
            risk_factors = {
                'volatility_risk': 'HIGH' if volatility > 0.03 else 'MEDIUM' if volatility > 0.015 else 'LOW',
                'trend_risk': 'LOW' if trend_strength > 25 else 'MEDIUM' if trend_strength > 15 else 'HIGH',
                'liquidity_risk': 'LOW',  # Assume good liquidity for major stocks
                'market_risk': 'MEDIUM'   # Default market risk
            }
            
            # Overall risk level
            high_risks = sum(1 for risk in risk_factors.values() if risk == 'HIGH')
            risk_level = 'HIGH' if high_risks >= 2 else 'MEDIUM' if high_risks >= 1 else 'LOW'
            
            return {
                'risk_level': risk_level,
                'confidence': 0.75,
                'factors': risk_factors,
                'warnings': ['High volatility environment'] if volatility > 0.03 else []
            }
            
        except Exception as e:
            logger.error(f"❌ Risk assessment error: {str(e)}")
            return {'risk_level': 'MEDIUM', 'confidence': 0.5, 'factors': {}, 'warnings': []}
    
    def _consensus_execution_plan(self, agent_views: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Create execution plan based on agent consensus"""
        try:
            executor_signals = agent_views.get('executor', {})
            
            return {
                'entry': executor_signals.get('entry', {'signal': False, 'type': 'manual_review'}),
                'exit': executor_signals.get('exit', {'signal': False, 'type': 'manual_review'}),
                'position_size': executor_signals.get('position_size', 0.05),
                'stop_loss': executor_signals.get('stop_loss'),
                'take_profit': executor_signals.get('take_profit')
            }
            
        except Exception as e:
            logger.error(f"❌ Execution plan error: {str(e)}")
            return {
                'entry': {'signal': False, 'type': 'manual_review'},
                'exit': {'signal': False, 'type': 'manual_review'},
                'position_size': 0.05,
                'stop_loss': None,
                'take_profit': None
            }
    
    def _generate_hedge_fund_recommendation(self, consensus: Dict[str, Any], data: pd.DataFrame, 
                                          ticker: str, current_price: float, regime: str = 'range') -> Dict[str, Any]:
        """
        GENERATE FINAL HEDGE FUND RECOMMENDATION
        
        Synthesizes all agent inputs into a unified recommendation that follows
        hedge fund best practices: risk management, position sizing, and clear exit strategies.
        """
        try:
            # Extract consensus components
            market_analysis = consensus['market_conditions']
            strategy = consensus['strategy_selection']
            risk_assessment = consensus['risk_assessment']
            execution_plan = consensus['execution_plan']
            
            # Perform probability fusion if vision analysis is available
            analyst_view = consensus.get('analyst_view', {})
            vision_analysis = consensus.get('vision_analysis', {})
            fused_probabilities = self.fuse_agent_probabilities(analyst_view, vision_analysis, regime)
            
            # Store fusion results in recommendation for metrics tracking
            recommendation_data = {
                'fusion_analysis': fused_probabilities,
                'regime': regime
            }
            
            # Determine final action with enhanced decision logic
            action = self._determine_final_action(consensus, market_analysis, strategy, regime, fused_probabilities)
            
            # Generate comprehensive recommendation
            recommendation = {
                'action': action,
                'market_analysis': market_analysis,
                'strategy': strategy,
                'signals': execution_plan,
                'risk_assessment': risk_assessment,
                'parameters': self._generate_trade_parameters(strategy, execution_plan, market_analysis),
                'consensus_score': consensus['agreement_score'],
                'consensus_details': {
                    'agreement_score': consensus.get('agreement_score', 0.0),
                    'consensus_reached': consensus.get('consensus_reached', False),
                    'threshold': consensus.get('threshold', self.consensus_threshold),
                    'conflicts': consensus.get('conflicts', []),
                    'final_decision': consensus.get('final_decision', action)
                },
                'fusion_analysis': fused_probabilities,  # Add fusion results
                'regime': regime,  # Add regime information
                'hedge_fund_notes': self._generate_hedge_fund_notes(consensus, action, strategy)
            }
            
            # Apply hedge fund risk controls
            recommendation = self._apply_risk_controls(recommendation, data, current_price)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Final recommendation error: {str(e)}")
            return self._generate_fallback_recommendation(data, ticker, current_price)
    
    def _determine_final_action(self, consensus: Dict[str, Any], market_analysis: Dict[str, Any], 
                               strategy: Dict[str, Any], regime: str = 'range', 
                               fused_probabilities: Dict[str, float] = None) -> str:
        """
        Determine final trading action using decision thresholds and no-trade zones
        """
        try:
            # Get fused probabilities if available
            if fused_probabilities:
                up_prob = fused_probabilities.get('up_probability', 0.5)
                inside_prob = fused_probabilities.get('inside_probability', 0.5)
            else:
                # Fallback to basic consensus
                up_prob = 0.6 if market_analysis.get('MACD_Signal') == 'bullish' else 0.4
                inside_prob = 0.5
            
            agreement_score = consensus.get('agreement_score', 0.0)
            confidence = strategy.get('confidence', 0.0)
            
            # Enhanced decision thresholds based on regime
            regime_thresholds = {
                'trend': {
                    'bullish_threshold': 0.58,
                    'bearish_threshold': 0.42,
                    'no_trade_lower': 0.47,
                    'no_trade_upper': 0.53,
                    'min_confidence': 0.6
                },
                'range': {
                    'bullish_threshold': 0.62,
                    'bearish_threshold': 0.38,
                    'no_trade_lower': 0.45,
                    'no_trade_upper': 0.55,
                    'min_confidence': 0.5
                },
                'event': {
                    'bullish_threshold': 0.65,
                    'bearish_threshold': 0.35,
                    'no_trade_lower': 0.40,
                    'no_trade_upper': 0.60,
                    'min_confidence': 0.7
                }
            }
            
            thresholds = regime_thresholds.get(regime, regime_thresholds['range'])
            
            # Additional market condition filters
            rsi = market_analysis.get('RSI', 50.0)
            trend_strength = market_analysis.get('trend_strength', 0.0)
            ivr = market_analysis.get('iv_rank', 0.0)
            
            # Check minimum agreement and confidence
            if agreement_score < 0.5 or confidence < thresholds['min_confidence']:
                logger.info(f"📋 NO TRADE: Insufficient consensus (agreement={agreement_score:.2f}, "
                           f"confidence={confidence:.2f})")
                return 'HOLD'
            
            # NO-TRADE ZONE: Avoid marginal probabilities
            if thresholds['no_trade_lower'] <= up_prob <= thresholds['no_trade_upper']:
                logger.info(f"📋 NO TRADE: Probability in no-trade zone ({up_prob:.3f})")
                return 'HOLD'
            
            # BULLISH CONDITIONS with regime-specific logic
            if up_prob >= thresholds['bullish_threshold']:
                # Additional bullish filters
                if regime == 'trend':
                    # In trending markets, require trend confirmation
                    if trend_strength > 20 and rsi > 25:  # Not deeply oversold
                        return 'BUY'
                elif regime == 'range':
                    # In ranging markets, favor mean reversion setups
                    if rsi < 65:  # Not overbought
                        return 'BUY'
                elif regime == 'event':
                    # In event-driven markets, require higher conviction
                    if confidence > 0.75 and agreement_score > 0.7:
                        return 'BUY'
                else:
                    return 'BUY'
            
            # BEARISH CONDITIONS with regime-specific logic
            elif up_prob <= thresholds['bearish_threshold']:
                # Additional bearish filters
                if regime == 'trend':
                    # In trending markets, require trend confirmation
                    if trend_strength > 20 and rsi < 75:  # Not deeply overbought
                        return 'SELL'
                elif regime == 'range':
                    # In ranging markets, favor mean reversion setups
                    if rsi > 35:  # Not oversold
                        return 'SELL'
                elif regime == 'event':
                    # In event-driven markets, require higher conviction
                    if confidence > 0.75 and agreement_score > 0.7:
                        return 'SELL'
                else:
                    return 'SELL'
            
            # OPTIONS-SPECIFIC DECISIONS
            strategy_name = strategy.get('name', '').lower()
            if 'iron condor' in strategy_name or 'strangle' in strategy_name:
                # Range-bound strategies need high inside probability
                if inside_prob >= 0.62:
                    logger.info(f"📊 RANGE STRATEGY: High inside probability ({inside_prob:.3f})")
                    return 'BUY'  # Enter the range strategy
            
            # IV RANK considerations for options
            if any(term in strategy_name for term in ['call', 'put', 'option']):
                if ivr <= 25:
                    # Low IV environment - prefer debit strategies
                    if 'call' in strategy_name or 'put' in strategy_name:
                        return 'BUY' if up_prob > 0.5 else 'SELL'
                elif ivr >= 35:
                    # High IV environment - prefer credit strategies  
                    if 'credit' in strategy_name or 'covered' in strategy_name:
                        return 'BUY' if up_prob > 0.5 else 'SELL'
            
            # DEFAULT: HOLD
            logger.info(f"📋 HOLD: Conditions not met for entry "
                       f"(up_prob={up_prob:.3f}, regime={regime})")
            return 'HOLD'
                
        except Exception as e:
            logger.error(f"❌ Action determination error: {str(e)}")
            return 'HOLD'
    
    def _generate_trade_parameters(self, strategy: Dict[str, Any], execution_plan: Dict[str, Any], 
                                 market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific trade parameters based on strategy and market conditions"""
        try:
            base_params = strategy.get('parameters', {})
            
            # Calculate concrete price values if not available
            current_price = market_analysis.get('current_price', 0)
            
            # Default values based on current price and ATR
            atr_value = market_analysis.get('ATR', current_price * 0.01)  # Default to 1% of price if ATR not available
            
            # Calculate stop loss and take profit if not provided
            stop_loss = execution_plan.get('stop_loss')
            if stop_loss is None and current_price > 0:
                if strategy.get('type') == 'long' or strategy.get('name', '').lower().find('call') >= 0:
                    stop_loss = current_price * 0.97  # 3% stop loss for long positions
                else:
                    stop_loss = current_price * 1.03  # 3% stop loss for short positions
                    
            take_profit = execution_plan.get('take_profit')
            if take_profit is None and current_price > 0:
                if strategy.get('type') == 'long' or strategy.get('name', '').lower().find('call') >= 0:
                    take_profit = current_price * 1.05  # 5% profit target for long positions
                else:
                    take_profit = current_price * 0.95  # 5% profit target for short positions
                    
            # Calculate position size based on risk
            position_size = execution_plan.get('position_size', 0.05)
            if position_size <= 0.1 and current_price > 0:
                # Calculate actual shares based on $10K account
                risk_per_share = abs(current_price - stop_loss) if stop_loss else current_price * 0.03
                if risk_per_share > 0:
                    # Base calculation on a standard $10,000 account with 2% max risk
                    position_size = round((10000 * 0.02) / risk_per_share, 1)
                    if position_size < 1:
                        position_size = 1  # Minimum 1 share
            
            # Add execution-specific parameters
            trade_params = {
                **base_params,
                'entry_condition': execution_plan.get('entry', {}).get('type', 'market_confirmation'),
                'exit_condition': execution_plan.get('exit', {}).get('type', 'profit_target_or_stop'),
                'position_size': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'max_holding_period': strategy.get('timeframe', 'short_term'),
                'volatility_adjustment': market_analysis.get('trend_strength', 0.0) > 25
            }
            
            return trade_params
            
        except Exception as e:
            logger.error(f"❌ Trade parameters error: {str(e)}")
            return {'entry_condition': 'manual_review', 'exit_condition': 'manual_review'}
    
    def _generate_hedge_fund_notes(self, consensus: Dict[str, Any], action: str, 
                                 strategy: Dict[str, Any]) -> List[str]:
        """Generate hedge fund style investment committee notes"""
        notes = []
        
        try:
            agreement_score = consensus.get('agreement_score', 0.0)
            
            # Consensus assessment
            if agreement_score >= 0.8:
                notes.append("STRONG CONSENSUS: All agents align on strategy and direction")
            elif agreement_score >= 0.6:
                notes.append("MODERATE CONSENSUS: Agents generally agree with minor differences")
            else:
                notes.append("LOW CONSENSUS: Significant disagreement between agents - proceed with caution")
            
            # Strategy rationale
            notes.append(f"PRIMARY STRATEGY: {strategy.get('name', 'Unknown')} selected based on current market conditions")
            
            # Risk management notes
            risk_level = strategy.get('risk_profile', 'MEDIUM')
            notes.append(f"RISK PROFILE: {risk_level} - Position sizing and stops adjusted accordingly")
            
            # Action justification
            if action == 'BUY':
                notes.append("RECOMMENDATION: LONG position justified by bullish technical confluence")
            elif action == 'SELL':
                notes.append("RECOMMENDATION: SHORT position justified by bearish technical confluence") 
            else:
                notes.append("RECOMMENDATION: HOLD - insufficient conviction for directional bet")
            
            # Add strategy-specific notes
            if strategy.get('name') == 'Iron Condors':
                notes.append("OPTIONS STRATEGY: Neutral outlook - profit from time decay and range-bound movement")
            elif 'Day Trading' in strategy.get('name', ''):
                notes.append("INTRADAY STRATEGY: High-frequency approach - requires active monitoring")
            
        except Exception as e:
            logger.error(f"❌ Hedge fund notes error: {str(e)}")
            notes.append("SYSTEM NOTE: Analysis completed with partial data - review recommended")
        
        return notes
    
    # Helper methods for consensus building and risk management
    def _enhance_strategy_parameters(self, strategy: Dict[str, Any], strategy_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance strategy parameters with trading_strategies.py data"""
        try:
            base_params = strategy.get('parameters', {})
            
            # Add strategy-specific parameters from our database
            enhanced_params = {
                **base_params,
                'timeframe': strategy_info.get('Timeframe', 'Unknown'),
                'risk_management': {
                    'pros': strategy_info.get('Pros', []),
                    'cons': strategy_info.get('Cons', []),
                    'when_to_use': strategy_info.get('When to Use', 'See documentation'),
                    'suitable_for': strategy_info.get('Suitable For', 'Experienced traders')
                }
            }
            
            return enhanced_params
            
        except Exception as e:
            logger.error(f"❌ Parameter enhancement error: {str(e)}")
            return strategy.get('parameters', {})
    
    def _assess_strategy_risk(self, strategy_info: Dict[str, Any], data: pd.DataFrame) -> str:
        """Assess risk level of selected strategy based on current market conditions"""
        try:
            strategy_name = strategy_info.get('Strategy', '')
            
            # High-risk strategies
            if any(term in strategy_name for term in ['Day Trading', 'Scalping']):
                return 'HIGH'
            
            # Medium-risk strategies  
            elif any(term in strategy_name for term in ['Swing Trading', 'Straddle', 'Strangle']):
                return 'MEDIUM'
            
            # Lower-risk strategies
            elif any(term in strategy_name for term in ['Covered Calls', 'Protective Puts', 'Iron Condors']):
                return 'LOW'
            
            else:
                return 'MEDIUM'  # Default
                
        except Exception as e:
            logger.error(f"❌ Strategy risk assessment error: {str(e)}")
            return 'MEDIUM'
    
    def _calculate_signal_agreement(self, signals: List[Any]) -> float:
        """Calculate agreement score for directional signals"""
        try:
            valid_signals = [s for s in signals if s is not None]
            if not valid_signals:
                return 0.5
            
            # Simple majority agreement
            signal_counts = {}
            for signal in valid_signals:
                signal_str = str(signal).lower()
                signal_counts[signal_str] = signal_counts.get(signal_str, 0) + 1
            
            if not signal_counts:
                return 0.5
                
            max_count = max(signal_counts.values())
            total_count = len(valid_signals)
            return max_count / total_count
            
        except Exception:
            return 0.5
    
    def _calculate_strategy_agreement(self, strategies: List[Any]) -> float:
        """Calculate agreement score for strategy preferences"""
        try:
            valid_strategies = [s for s in strategies if s is not None]
            if not valid_strategies:
                return 0.5
            
            # Count most common strategy
            from collections import Counter
            strategy_counts = Counter(valid_strategies)
            most_common_count = strategy_counts.most_common(1)[0][1] if strategy_counts else 0
            
            return most_common_count / len(valid_strategies) if valid_strategies else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_risk_agreement(self, risk_levels: List[Any]) -> float:
        """Calculate agreement score for risk assessments"""
        try:
            valid_risks = [r for r in risk_levels if r is not None]
            if not valid_risks:
                return 0.5
            
            # Convert to numeric and check variance
            risk_numeric = []
            for risk in valid_risks:
                if str(risk).upper() == 'LOW':
                    risk_numeric.append(1)
                elif str(risk).upper() == 'MEDIUM':
                    risk_numeric.append(2) 
                elif str(risk).upper() == 'HIGH':
                    risk_numeric.append(3)
            
            if not risk_numeric:
                return 0.5
            
            # Low variance = high agreement
            variance = np.var(risk_numeric) if len(risk_numeric) > 1 else 0
            return max(0.0, 1.0 - variance/2.0)  # Scale variance to agreement score
            
        except Exception:
            return 0.5
    
    def _identify_conflicts(self, agent_views: Dict[str, Any]) -> List[str]:
        """Identify specific conflicts between agents"""
        conflicts = []
        
        try:
            # Check for directional conflicts
            analyst_direction = agent_views.get('analyst', {}).get('trend', {}).get('direction')
            strategist_type = agent_views.get('strategist', {}).get('type') 
            executor_signal = agent_views.get('executor', {}).get('entry', {}).get('type')
            
            directions = [analyst_direction, strategist_type, executor_signal]
            bullish_signals = sum(1 for d in directions if str(d).lower() in ['bullish', 'momentum', 'buy'])
            bearish_signals = sum(1 for d in directions if str(d).lower() in ['bearish', 'mean_reversion', 'sell'])
            
            if bullish_signals > 0 and bearish_signals > 0:
                conflicts.append("Directional disagreement between agents")
            
            # Check for strategy conflicts
            strategies = [
                agent_views.get('strategist', {}).get('name'),
                agent_views.get('analyst', {}).get('recommended_strategy'),
                agent_views.get('executor', {}).get('preferred_strategy')
            ]
            unique_strategies = set(s for s in strategies if s is not None)
            if len(unique_strategies) > 1:
                conflicts.append("Strategy preference disagreement")
            
        except Exception as e:
            logger.error(f"❌ Conflict identification error: {str(e)}")
            conflicts.append("Unable to assess conflicts")
        
        return conflicts
    
    def _resolve_conflicts(self, consensus: Dict[str, Any], agent_views: Dict[str, Any], 
                          data: pd.DataFrame) -> Dict[str, Any]:
        """Resolve conflicts using hierarchy and additional analysis"""
        try:
            # Use strategist as primary decision maker for strategy selection
            primary_strategy = agent_views.get('strategist', {}).get('name', 'Day Trading Calls/Puts')
            
            # Use analyst for market direction
            market_direction = agent_views.get('analyst', {}).get('trend', {}).get('direction', 'neutral')
            
            # Update consensus with conflict resolution
            consensus['strategy_selection']['name'] = primary_strategy
            consensus['market_conditions']['primary_trend'] = market_direction
            consensus['conflict_resolution'] = 'Strategist priority for strategy, Analyst priority for direction'
            
            # Increase confidence slightly after resolution
            consensus['agreement_score'] = min(0.7, consensus['agreement_score'] + 0.1)
            
        except Exception as e:
            logger.error(f"❌ Conflict resolution error: {str(e)}")
        
        return consensus
    
    def _make_final_decision(self, consensus: Dict[str, Any], agent_views: Dict[str, Any]) -> str:
        """Make final investment committee decision"""
        try:
            agreement_score = consensus.get('agreement_score', 0.0)
            market_analysis = consensus.get('market_conditions', {})
            
            # High agreement threshold for action
            if agreement_score >= 0.7:
                macd_signal = market_analysis.get('MACD_Signal', 'neutral')
                if macd_signal == 'bullish':
                    return 'BUY'
                elif macd_signal == 'bearish':
                    return 'SELL'
                else:
                    return 'HOLD'
            else:
                return 'HOLD'  # Conservative default
                
        except Exception as e:
            logger.error(f"❌ Final decision error: {str(e)}")
            return 'HOLD'
    
    def _apply_risk_controls(self, recommendation: Dict[str, Any], data: pd.DataFrame, 
                           current_price: float) -> Dict[str, Any]:
        """Apply hedge fund risk management controls"""
        try:
            # Position sizing based on volatility
            volatility = data['ATR'].iloc[-1] / current_price if 'ATR' in data and len(data) > 0 else 0.02
            risk_limits = getattr(self, 'risk_limits', {'max_position_size': 0.2})
            max_position = min(risk_limits.get('max_position_size', 0.2), 0.02 / volatility)
            
            # Set default parameters if they don't exist
            if 'parameters' not in recommendation:
                recommendation['parameters'] = {}
                
            # Calculate entry price, stop loss, and take profit if not set
            if 'entry_price' not in recommendation['parameters'] or not recommendation['parameters']['entry_price']:
                recommendation['parameters']['entry_price'] = current_price
                
            if 'stop_loss' not in recommendation['parameters'] or not recommendation['parameters']['stop_loss']:
                # Default stop loss based on 2x ATR
                atr = data['ATR'].iloc[-1] if 'ATR' in data and len(data) > 0 else current_price * 0.02
                if recommendation['action'] == 'BUY':
                    recommendation['parameters']['stop_loss'] = current_price - (atr * 2)
                else:
                    recommendation['parameters']['stop_loss'] = current_price + (atr * 2)
                    
            if 'take_profit' not in recommendation['parameters'] or not recommendation['parameters']['take_profit']:
                # Default take profit based on 3x ATR
                atr = data['ATR'].iloc[-1] if 'ATR' in data and len(data) > 0 else current_price * 0.02
                if recommendation['action'] == 'BUY':
                    recommendation['parameters']['take_profit'] = current_price + (atr * 3)
                else:
                    recommendation['parameters']['take_profit'] = current_price - (atr * 3)
            
            # Update position size in parameters
            recommendation['parameters']['position_size'] = max_position
            recommendation['parameters']['volatility_adjusted'] = True
            
            # Calculate risk metrics
            stop_loss = recommendation['parameters']['stop_loss']
            if stop_loss and current_price > 0:
                # Calculate max loss percentage
                if recommendation['action'] == 'BUY':
                    max_loss_pct = (current_price - stop_loss) / current_price * 100
                else:
                    max_loss_pct = (stop_loss - current_price) / current_price * 100
                
                # Add risk assessment
                if 'risk_assessment' not in recommendation:
                    recommendation['risk_assessment'] = {}
                    
                recommendation['risk_assessment']['max_loss'] = abs(max_loss_pct)
                
                # Calculate portfolio risk assuming $10K account
                position_shares = max_position / current_price * 10000
                max_dollar_risk = position_shares * abs(current_price - stop_loss)
                portfolio_risk_pct = (max_dollar_risk / 10000) * 100
                recommendation['risk_assessment']['portfolio_risk'] = portfolio_risk_pct
            
            # Add risk controls
            recommendation['risk_controls'] = {
                'max_position_size': max_position,
                'volatility_limit': volatility,
                'stop_loss_required': True,
                'max_holding_period': '1 week'
            }
            
        except Exception as e:
            logger.error(f"❌ Risk controls error: {str(e)}")
        
        return recommendation
    
    def _generate_fallback_recommendation(self, data: pd.DataFrame, ticker: str, current_price: float) -> Dict[str, Any]:
        """Generate conservative fallback recommendation when analysis fails"""
        return {
            'action': 'HOLD',
            'market_analysis': {
                'RSI': data['RSI'].iloc[-1] if 'RSI' in data and len(data) > 0 else 50.0,
                'MACD_Signal': 'neutral',
                'volume_signal': 'unknown',
                'trend_strength': 0.0
            },
            'strategy': {
                'name': 'Day Trading Calls/Puts',
                'confidence': 0.3,
                'type': 'conservative',
                'parameters': {'manual_review_required': True}
            },
            'signals': {'entry': {'signal': False}, 'exit': {'signal': False}},
            'risk_assessment': {'risk_level': 'HIGH', 'confidence': 0.3},
            'parameters': {'manual_review_required': True},
            'consensus_score': 0.0,
            'hedge_fund_notes': ['FALLBACK MODE: Manual review required due to analysis errors']
        }
        
    # Legacy method for backwards compatibility
    def analyze_market(self, data: pd.DataFrame, ticker: str, fundamental_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Legacy method - redirects to new analyze_and_recommend"""
        current_price = data['Close'].iloc[-1] if len(data) > 0 else 0.0
        return self.analyze_and_recommend(data, ticker, current_price)
