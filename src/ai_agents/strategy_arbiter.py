import json
from typing import List, Dict, Any, Optional

def score_strategy(candidate: Dict[str, Any], user_timeframe: str, features: Dict[str, Any]) -> int:
    """Score a candidate strategy based on fit, regime, volatility, and signal consistency."""
    s = 0
    # Timeframe fit (40%)
    s += 40 if candidate.get('timeframe') == user_timeframe else 0
    # Regime alignment (25%)
    s += 25 if regime_matches(candidate, features) else 0
    # Volatility fit (20%)
    s += 20 if iv_fits(candidate, features.get('iv_rank', 0)) else 0
    # Signal consistency (15%)
    s += 15 if signals_are_consistent(features) else 0
    return s

def regime_matches(candidate: Dict[str, Any], features: Dict[str, Any]) -> bool:
    # Example: match trend direction and ADX
    trend = features.get('trend')
    adx = features.get('adx', 0)
    if candidate.get('trend') and candidate['trend'] != trend:
        return False
    if candidate.get('requires_trend') and adx < 18:
        return False
    return True

def iv_fits(candidate: Dict[str, Any], iv_rank: float) -> bool:
    # Example: credit spreads need high IV, buying premium needs low IV
    if candidate.get('type') == 'credit_spread' and iv_rank < 20:
        return False
    if candidate.get('type') == 'buy_premium' and iv_rank > 60:
        return False
    return True

def signals_are_consistent(features: Dict[str, Any]) -> bool:
    # Example: penalize if RSI and MACD disagree
    rsi = features.get('rsi', 50)
    macd = features.get('macd', 0)
    if (rsi < 40 and macd > 0) or (rsi > 60 and macd < 0):
        return False
    return True

def short_reason(candidate: Dict[str, Any], features: Dict[str, Any]) -> str:
    # Give a short reason why this candidate was not selected
    if candidate.get('timeframe') != features.get('user_timeframe'):
        return 'Timeframe mismatch'
    if not regime_matches(candidate, features):
        return 'Regime/Trend mismatch'
    if not iv_fits(candidate, features.get('iv_rank', 0)):
        return 'IV/volatility mismatch'
    if not signals_are_consistent(features):
        return 'Mixed signals'
    return 'Lower score'

def choose_final_strategy(candidates: List[Dict[str, Any]], user_timeframe: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the best strategy and provide alternatives with reasons.
    """
    if not candidates:
        return {"error": "No strategies provided"}
    features = dict(features)
    features['user_timeframe'] = user_timeframe
    ranked = sorted(candidates, key=lambda c: score_strategy(c, user_timeframe, features), reverse=True)
    final = ranked[0]
    alternatives = []
    for alt in ranked[1:2]:
        alternatives.append({
            "strategy_name": alt.get('name', 'Unknown'),
            "why_not": short_reason(alt, features)
        })
    final['alternatives'] = alternatives
    return final
