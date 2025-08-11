# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

def calculate_volatility_regime(data: pd.DataFrame, window: int = 20) -> str:
    """
    Calculate the volatility regime of a financial instrument.
    
    Args:
        data: DataFrame containing price data
        window: The window size for volatility calculation
        
    Returns:
        str: The volatility regime ('high_volatility', 'normal_volatility', 'low_volatility')
    """
    try:
        # Calculate historical volatility (standard deviation of returns)
        if len(data) < window:
            return "normal_volatility"
            
        returns = data['Close'].pct_change().dropna()
        if len(returns) == 0:
            return "normal_volatility"
            
        current_vol = returns.iloc[-window:].std() * np.sqrt(252)  # Annualized
        historical_vol = returns.iloc[:-window].std() * np.sqrt(252) if len(returns) > window*2 else current_vol
        
        # Determine volatility regime
        if current_vol > historical_vol * 1.5:
            return "high_volatility"
        elif current_vol < historical_vol * 0.5:
            return "low_volatility"
        else:
            return "normal_volatility"
            
    except Exception as e:
        logger.error(f"Error in volatility regime calculation: {str(e)}")
        return "normal_volatility"

def detect_market_regime(data: pd.DataFrame) -> str:
    """
    Detects the current market regime based on price action and indicators.
    
    Args:
        data: DataFrame with OHLCV and indicator data
        
    Returns:
        str: Market regime ('bullish_trend', 'bearish_trend', 'bullish_reversal',
             'bearish_reversal', 'sideways_volatility', 'sideways_low_volatility')
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        logger.error("Invalid or empty input data")
        return "unknown"
    
    try:
        # Extract needed indicators (with flexible column names)
        close_prices = data['Close'].values
        
        # Get RSI with flexible column naming
        rsi = None
        for rsi_col in ['RSI', 'RSI_14', 'RSI_1d']:
            if rsi_col in data.columns:
                rsi = data[rsi_col].iloc[-1]
                break
        
        # Get ADX with flexible column naming
        adx = None
        for adx_col in ['ADX', 'ADX_14', 'ADX_1d']:
            if adx_col in data.columns:
                adx = data[adx_col].iloc[-1]
                break
        
        # Get MACD with flexible column naming
        macd = None
        macd_signal = None
        for macd_col in ['MACD', 'MACD_1d']:
            if macd_col in data.columns:
                signal_col = f"{macd_col.split('_')[0]}_Signal" if '_' in macd_col else 'MACD_Signal'
                if signal_col in data.columns:
                    macd = data[macd_col].iloc[-1]
                    macd_signal = data[signal_col].iloc[-1]
                    break
                
        # Get Bollinger Band width
        bb_width = None
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns and 'BB_middle' in data.columns:
            bb_upper = data['BB_upper'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            bb_middle = data['BB_middle'].iloc[-1]
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
        
        # Calculate trend direction (using SMA20 vs SMA50 relationship)
        # Make sure indicators are not None and columns exist
        sma20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]) else None
        sma50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns and not pd.isna(data['SMA_50'].iloc[-1]) else None
        
        # Calculate price momentum (rate of change)
        momentum = data['Close'].pct_change(5).iloc[-1] if len(data) > 5 else 0
        
        # Calculate volatility
        volatility = data['Close'].pct_change().rolling(window=20).std().iloc[-1] if len(data) > 20 else 0
        
        # Detect regime based on signals
        
        # Strong trend indicators
        trending = adx > 25 if adx is not None else False
        strong_trend = adx > 35 if adx is not None else False
        
        # Trend direction indicators
        bullish_trend = False
        if sma20 is not None and sma50 is not None:
            bullish_trend = sma20 > sma50 and close_prices[-1] > sma20
        
        # Overbought/oversold indicators
        overbought = rsi > 70 if rsi is not None else False
        oversold = rsi < 30 if rsi is not None else False
        
        # Momentum indicators
        bullish_momentum = macd > macd_signal if macd is not None and macd_signal is not None else False
        
        # Volatility indicators
        high_volatility = bb_width > 0.05 if bb_width is not None else volatility > 0.02
        low_volatility = bb_width < 0.02 if bb_width is not None else volatility < 0.01
        
        # Regime classification - matching the regimes used in predict_next_day_close
        if bullish_trend:
            if high_volatility:
                return "volatile_bullish"
            else:
                return "trending_bullish"
        elif not bullish_trend and trending:
            if high_volatility:
                return "volatile_bearish"
            else:
                return "trending_bearish"
        elif low_volatility:
            return "range_bound"
        else:
            # Default case for undefined regimes
            return "range_bound"
            
    except Exception as e:
        logger.error(f"Error in market regime detection: {str(e)}")
        return "unknown"

def volatility_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate volatility-specific features for different market regimes.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        DataFrame with additional volatility-based features
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data
        
    try:
        df = data.copy()
        
        # Calculate basic volatility metrics with safety checks
        if 'Close' in df.columns:
            # Historical volatility (standard deviation of returns)
            df['Returns'] = df['Close'].pct_change().fillna(0)
            
            # Apply safety checks to rolling calculations
            vol_20d = df['Returns'].rolling(window=20).std()
            vol_10d = df['Returns'].rolling(window=10).std()
            
            # Make sure to handle NaN values
            df['Volatility_20d'] = vol_20d.fillna(0) * np.sqrt(252)  # Annualized
            df['Volatility_10d'] = vol_10d.fillna(0) * np.sqrt(252)
            
            # Volatility ratio (short-term vs long-term) with safety check
            df['Vol_Ratio'] = df['Volatility_10d'] / (df['Volatility_20d'] + 1e-5)
            df['Vol_Ratio'] = df['Vol_Ratio'].fillna(1.0)  # Default to 1.0 for missing values
            
            # Volatility trend with safety check
            vol_trend = df['Volatility_20d'].pct_change(5)
            df['Vol_Trend'] = vol_trend.fillna(0)
            
            # Extreme moves with safety check
            df['Extreme_Move'] = (df['Returns'].abs() > df['Volatility_20d'] / np.sqrt(252) * 2).astype(int)
            
            # Consecutive extreme moves with safety check
            extreme_streak = df['Extreme_Move'].rolling(window=5).sum()
            df['Extreme_Streak'] = extreme_streak.fillna(0).astype(int)
            
        # ATR-based features if available
        if 'ATR' in df.columns and 'Close' in df.columns:
            # ATR Percentage
            df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
            
            # ATR Trend
            df['ATR_Trend'] = df['ATR'].pct_change(5)
            
            # ATR Percentile (current ATR vs historical)
            df['ATR_Percentile'] = df['ATR'].rolling(window=50).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] 
                if len(x) == 50 else np.nan
            )
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating volatility features: {str(e)}")
        return data

def generate_regime_specific_features(data: pd.DataFrame, market_regime: str = None) -> pd.DataFrame:
    """
    Generate regime-specific features to enhance prediction accuracy.
    
    Args:
        data: DataFrame with price and indicator data
        market_regime: Optional pre-detected market regime. If None, will detect for each row.
        
    Returns:
        DataFrame with additional regime-specific features
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data
    
    try:
        df = data.copy()
        
        # Detect market regime for each bar or use provided regime
        if market_regime is None:
            regimes = []
            for i in range(len(df)):
                if i < 50:  # Need enough history for regime detection
                    regimes.append("unknown")
                else:
                    try:
                        regime = detect_market_regime(df.iloc[:i+1])
                        regimes.append(regime)
                    except Exception:
                        # If regime detection fails, use a default regime
                        regimes.append("unknown")
            
            df['Market_Regime'] = regimes
        else:
            # Use the provided market regime for all rows
            df['Market_Regime'] = market_regime
            
        # Ensure market regime is not None - safeguard
        df['Market_Regime'] = df['Market_Regime'].fillna("unknown")
        
        try:
            # One-hot encode regime for modeling
            regime_dummies = pd.get_dummies(df['Market_Regime'], prefix='Regime')
            df = pd.concat([df, regime_dummies], axis=1)
        except Exception as e:
            logger.warning(f"Error creating regime dummies: {str(e)}")
            # Continue without one-hot encoding
        
        # Generate regime-specific features with safety checks
        
        # 1. Trend following features - more important in trending regimes
        bullish_regimes = df['Market_Regime'].isin(['trending_bullish', 'volatile_bullish']).fillna(False)
        bearish_regimes = df['Market_Regime'].isin(['trending_bearish', 'volatile_bearish']).fillna(False)
        
        if 'SMA_20' in df.columns:
            # Add null safety checks
            sma_trend_bullish = df['Close'] > df['SMA_20']
            sma_trend_bearish = df['Close'] < df['SMA_20']
            
            # Handle NaN values safely
            df['Trend_Strength_Bullish'] = sma_trend_bullish.fillna(False).astype(int) * bullish_regimes.astype(int)
            df['Trend_Strength_Bearish'] = sma_trend_bearish.fillna(False).astype(int) * bearish_regimes.astype(int)
        
        # 2. Mean reversion features - more important in overbought/oversold regimes
        if 'RSI' in df.columns:
            # Make sure RSI values exist and are valid
            if df['RSI'].notna().any():
                # In range-bound regimes, mean reversion is important
                rsi_high = (df['RSI'] > 70).fillna(False)
                rsi_low = (df['RSI'] < 30).fillna(False)
                range_bound = (df['Market_Regime'] == 'range_bound').fillna(False)
                
                df['Mean_Reversion_Signal'] = ((rsi_high * -1) + (rsi_low * 1)) * range_bound.astype(int)
            else:
                df['Mean_Reversion_Signal'] = 0
        
        # 3. Volatility-adjusted features - more important in high volatility regimes
        high_vol_regimes = df['Market_Regime'].isin(['volatile_bullish', 'volatile_bearish']).fillna(False)
        if 'ATR' in df.columns and 'Close' in df.columns:
            try:
                # Check for valid ATR values
                if df['ATR'].notna().any():
                    # Normalize returns by ATR with safety checks
                    returns = df['Close'].pct_change().fillna(0)
                    atr_ratio = (df['ATR'] / df['Close'] + 1e-5).fillna(1e-5)
                    
                    # Safe division with fillna for any resulting NaN values
                    df['Returns_Vol_Adjusted'] = (returns / atr_ratio).fillna(0)
                    
                    # Higher weight in volatile regimes
                    df['Returns_Vol_Regime'] = df['Returns_Vol_Adjusted'] * high_vol_regimes.astype(int)
                else:
                    df['Returns_Vol_Adjusted'] = 0
                    df['Returns_Vol_Regime'] = 0
            except Exception as e:
                logger.warning(f"Error calculating volatility-adjusted returns: {str(e)}")
                df['Returns_Vol_Adjusted'] = 0
                df['Returns_Vol_Regime'] = 0
        
        # 4. Momentum features - more important in trending regimes
        trending_regimes = df['Market_Regime'].isin(['trending_bullish', 'trending_bearish']).fillna(False)
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            try:
                # Ensure both MACD values exist and are valid
                if df['MACD'].notna().any() and df['MACD_Signal'].notna().any():
                    macd_diff = (df['MACD'] - df['MACD_Signal']).fillna(0)
                    df['MACD_Trend_Signal'] = np.sign(macd_diff) * trending_regimes.astype(int)
                else:
                    df['MACD_Trend_Signal'] = 0
            except Exception as e:
                logger.warning(f"Error calculating MACD trend signal: {str(e)}")
                df['MACD_Trend_Signal'] = 0
        
        # 5. Consolidation breakout features - for range-bound regimes
        range_bound_regime = (df['Market_Regime'] == 'range_bound').fillna(False)
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns and 'Close' in df.columns:
            try:
                # Make sure BB values exist and are valid
                if df['BB_upper'].notna().any() and df['BB_lower'].notna().any():
                    above_upper = (df['Close'] > df['BB_upper']).fillna(False)
                    below_lower = (df['Close'] < df['BB_lower']).fillna(False)
                    
                    # Calculate breakout signal safely
                    df['Breakout_Signal'] = ((above_upper * 1) + (below_lower * -1)) * range_bound_regime.astype(int)
                else:
                    df['Breakout_Signal'] = 0
            except Exception as e:
                logger.warning(f"Error calculating breakout signal: {str(e)}")
                df['Breakout_Signal'] = 0
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating regime features: {str(e)}")
        return data
