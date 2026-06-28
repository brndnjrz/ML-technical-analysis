"""
Options strike and expiry optimization using expected value calculations.
Implements grid scoring without requiring live options chain data.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class OptionsGridScorer:
    """Score options strategies across strike/expiry combinations"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate assumption
        self.transaction_costs = {
            'stock': 0.005,  # 0.5% per side
            'options': 0.65,  # $0.65 per contract
            'slippage': 0.002  # 0.2% slippage
        }
    
    def score_option_grid(self, 
                         current_price: float, 
                         implied_vol: float,
                         up_probability: float,
                         inside_probability: float,
                         strategy_type: str,
                         days_to_expiry_range: Tuple[int, int] = (5, 45),
                         strike_range_pct: float = 0.15) -> Dict[str, Any]:
        """
        Score options strategies across a grid of strikes and expiries
        
        Args:
            current_price: Current stock price
            implied_vol: Implied volatility (annualized)
            up_probability: Probability of upward movement
            inside_probability: Probability price stays in range
            strategy_type: Type of strategy (calls, puts, iron_condor, etc.)
            days_to_expiry_range: Min and max days to expiry
            strike_range_pct: Strike range as % of current price
        
        Returns:
            Dict with best strategy parameters and scores
        """
        try:
            # Generate candidate strikes and expiries
            strikes = self._generate_strikes(current_price, strike_range_pct)
            expiries = self._generate_expiries(days_to_expiry_range)
            
            # Score all combinations
            best_score = float('-inf')
            best_candidate = None
            all_candidates = []
            
            for dte in expiries:
                for strike in strikes:
                    try:
                        # Calculate expected value for this combination
                        candidate = self._score_candidate(
                            strategy_type, current_price, strike, dte,
                            implied_vol, up_probability, inside_probability
                        )
                        
                        if candidate and candidate['expected_pnl'] > best_score:
                            best_score = candidate['expected_pnl']
                            best_candidate = candidate
                        
                        if candidate:
                            all_candidates.append(candidate)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Scoring error for strike ${strike:.2f}, {dte}DTE: {e}")
                        continue
            
            if best_candidate:
                # Add runner-ups for comparison
                sorted_candidates = sorted(all_candidates, 
                                         key=lambda x: x['expected_pnl'], reverse=True)
                
                result = {
                    'best_strategy': best_candidate,
                    'runner_up': sorted_candidates[1] if len(sorted_candidates) > 1 else None,
                    'total_candidates_scored': len(all_candidates),
                    'score_distribution': {
                        'mean': np.mean([c['expected_pnl'] for c in all_candidates]),
                        'std': np.std([c['expected_pnl'] for c in all_candidates]),
                        'best_score': best_score
                    }
                }
                
                logger.info(f"✅ Options grid scored: {len(all_candidates)} candidates, "
                           f"best expected PnL: ${best_score:.2f}")
                return result
            else:
                logger.error("❌ No valid option candidates found")
                return {'error': 'No valid candidates'}
                
        except Exception as e:
            logger.error(f"❌ Options grid scoring error: {e}")
            return {'error': str(e)}
    
    def _generate_strikes(self, current_price: float, range_pct: float) -> List[float]:
        """Generate realistic strike prices"""
        # Create strikes from -range_pct to +range_pct in $1 increments for stocks > $20
        # or $0.50 increments for stocks < $20
        increment = 0.5 if current_price < 20 else 1.0
        
        lower_bound = current_price * (1 - range_pct)
        upper_bound = current_price * (1 + range_pct)
        
        strikes = []
        strike = lower_bound
        while strike <= upper_bound:
            strikes.append(round(strike / increment) * increment)
            strike += increment
        
        return list(set(strikes))  # Remove duplicates
    
    def _generate_expiries(self, days_range: Tuple[int, int]) -> List[int]:
        """Generate realistic expiry dates (in days)"""
        min_dte, max_dte = days_range
        
        # Focus on common expiry dates: weekly and monthly
        expiries = []
        
        # Weekly expiries (every 7 days)
        for dte in range(7, min(max_dte + 1, 35), 7):
            if dte >= min_dte:
                expiries.append(dte)
        
        # Monthly expiries (approximately)
        for dte in [30, 45, 60]:
            if min_dte <= dte <= max_dte:
                expiries.append(dte)
        
        return sorted(list(set(expiries)))
    
    def _score_candidate(self, strategy_type: str, current_price: float, 
                        strike: float, dte: int, implied_vol: float,
                        up_probability: float, inside_probability: float) -> Optional[Dict[str, Any]]:
        """Score an individual options strategy candidate"""
        
        try:
            # Calculate expected moves
            time_to_expiry = dte / 365.0
            expected_move = implied_vol * np.sqrt(time_to_expiry) * current_price
            
            # Generate scenarios: up, down, flat with probabilities
            scenarios = [
                {
                    'price': current_price + expected_move * 0.8,
                    'probability': up_probability,
                    'label': 'up'
                },
                {
                    'price': current_price - expected_move * 0.8,
                    'probability': 1 - up_probability,
                    'label': 'down'
                },
                {
                    'price': current_price,
                    'probability': inside_probability if strategy_type in ['iron_condor', 'strangle'] else 0.2,
                    'label': 'flat'
                }
            ]
            
            # Normalize probabilities
            total_prob = sum(s['probability'] for s in scenarios)
            for scenario in scenarios:
                scenario['probability'] /= total_prob
            
            # Calculate P&L for each scenario
            expected_pnl = 0
            for scenario in scenarios:
                pnl = self._calculate_strategy_pnl(
                    strategy_type, strike, scenario['price'], current_price, dte, implied_vol
                )
                expected_pnl += pnl * scenario['probability']
            
            # Subtract transaction costs
            transaction_cost = self._calculate_transaction_costs(strategy_type)
            net_expected_pnl = expected_pnl - transaction_cost
            
            # Calculate win probability (scenarios where P&L > 0)
            win_prob = sum(
                scenario['probability'] for scenario in scenarios
                if self._calculate_strategy_pnl(strategy_type, strike, scenario['price'], 
                                              current_price, dte, implied_vol) > 0
            )
            
            # Calculate maximum risk
            max_risk = self._calculate_max_risk(strategy_type, strike, current_price)
            
            # Risk-adjusted score (expected PnL / max risk)
            risk_adjusted_score = net_expected_pnl / max_risk if max_risk > 0 else net_expected_pnl
            
            return {
                'strategy_type': strategy_type,
                'strike': strike,
                'days_to_expiry': dte,
                'expected_pnl': net_expected_pnl,
                'win_probability': win_prob,
                'max_risk': max_risk,
                'risk_adjusted_score': risk_adjusted_score,
                'transaction_cost': transaction_cost,
                'scenarios_pnl': [
                    {
                        'scenario': s['label'],
                        'price': s['price'],
                        'probability': s['probability'],
                        'pnl': self._calculate_strategy_pnl(strategy_type, strike, s['price'], current_price, dte, implied_vol)
                    }
                    for s in scenarios
                ]
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Candidate scoring error: {e}")
            return None
    
    def _calculate_strategy_pnl(self, strategy_type: str, strike: float, 
                               final_price: float, current_price: float,
                               dte: int, implied_vol: float) -> float:
        """Calculate P&L for specific strategy at expiry"""
        
        # Simplified P&L calculations (assuming European-style expiry)
        if strategy_type.lower() in ['call', 'long_call']:
            # Long call P&L
            option_premium = self._estimate_option_price(current_price, strike, dte, implied_vol, 'call')
            intrinsic_value = max(0, final_price - strike)
            return intrinsic_value - option_premium
            
        elif strategy_type.lower() in ['put', 'long_put']:
            # Long put P&L
            option_premium = self._estimate_option_price(current_price, strike, dte, implied_vol, 'put')
            intrinsic_value = max(0, strike - final_price)
            return intrinsic_value - option_premium
            
        elif strategy_type.lower() == 'covered_call':
            # Covered call = Long stock + Short call
            stock_pnl = final_price - current_price
            call_premium = self._estimate_option_price(current_price, strike, dte, implied_vol, 'call')
            call_pnl = call_premium - max(0, final_price - strike)  # Short call P&L
            return stock_pnl + call_pnl
            
        elif strategy_type.lower() == 'protective_put':
            # Protective put = Long stock + Long put
            stock_pnl = final_price - current_price
            put_premium = self._estimate_option_price(current_price, strike, dte, implied_vol, 'put')
            put_pnl = max(0, strike - final_price) - put_premium  # Long put P&L
            return stock_pnl + put_pnl
            
        elif strategy_type.lower() in ['iron_condor', 'short_strangle']:
            # Simplified iron condor: profit if price stays near current level
            # Assume strikes are equidistant around current price
            wing_width = abs(strike - current_price)
            if abs(final_price - current_price) <= wing_width * 0.5:
                return wing_width * 0.3  # Profit in the middle
            else:
                return -wing_width * 0.7  # Loss at the wings
        
        else:
            # Fallback for unknown strategies
            return 0.0
    
    def _estimate_option_price(self, current_price: float, strike: float, 
                              dte: int, implied_vol: float, option_type: str) -> float:
        """Estimate option price using simplified Black-Scholes"""
        try:
            time_to_expiry = dte / 365.0
            
            if time_to_expiry <= 0:
                # At expiry, only intrinsic value
                if option_type == 'call':
                    return max(0, current_price - strike)
                else:
                    return max(0, strike - current_price)
            
            # Simplified Black-Scholes approximation
            d1 = (np.log(current_price / strike) + 
                  (self.risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry) / \
                 (implied_vol * np.sqrt(time_to_expiry))
            
            d2 = d1 - implied_vol * np.sqrt(time_to_expiry)
            
            if option_type == 'call':
                option_price = (current_price * norm.cdf(d1) - 
                              strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:  # put
                option_price = (strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                              current_price * norm.cdf(-d1))
            
            return max(option_price, 0.01)  # Minimum $0.01
            
        except Exception as e:
            logger.warning(f"⚠️ Option pricing error: {e}")
            # Fallback to simple intrinsic + time value
            intrinsic = max(0, (current_price - strike) if option_type == 'call' else (strike - current_price))
            time_value = implied_vol * current_price * np.sqrt(dte / 365) * 0.4
            return intrinsic + time_value
    
    def _calculate_transaction_costs(self, strategy_type: str) -> float:
        """Calculate estimated transaction costs for strategy"""
        base_cost = self.transaction_costs['options']
        
        if strategy_type.lower() in ['call', 'put']:
            return base_cost  # Single option
        elif strategy_type.lower() in ['covered_call', 'protective_put']:
            return base_cost + (self.transaction_costs['stock'] * 100)  # Option + 100 shares
        elif strategy_type.lower() in ['iron_condor', 'strangle']:
            return base_cost * 2  # Two options minimum
        else:
            return base_cost
    
    def _calculate_max_risk(self, strategy_type: str, strike: float, current_price: float) -> float:
        """Calculate maximum risk for the strategy"""
        if strategy_type.lower() in ['call', 'put']:
            # Maximum risk is the premium paid (estimated as 2-5% of stock price)
            return current_price * 0.03
        elif strategy_type.lower() == 'covered_call':
            # Risk is unlimited downside on stock
            return current_price  # Worst case: stock goes to zero
        elif strategy_type.lower() == 'protective_put':
            # Risk is limited by put strike
            return max(0, current_price - strike)
        elif strategy_type.lower() in ['iron_condor', 'short_strangle']:
            # Risk is the width of the wings
            wing_width = abs(strike - current_price)
            return wing_width
        else:
            return current_price * 0.05  # Default 5%

# Global scorer instance
options_scorer = OptionsGridScorer()

def optimize_options_strategy(current_price: float, implied_vol: float,
                            up_probability: float, inside_probability: float,
                            strategy_type: str) -> Dict[str, Any]:
    """Convenience function for options optimization"""
    return options_scorer.score_option_grid(
        current_price, implied_vol, up_probability, 
        inside_probability, strategy_type
    )
