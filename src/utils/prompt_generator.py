"""Prompt generation utilities for AI analysis."""


def generate_analysis_prompt(analysis_type, strategy_type, market_context, interval, ticker):
    """Generate appropriate prompt based on analysis type and strategy."""
    
    # Initialize default prompt
    default_prompt = f"""
    You are an expert financial analyst. Analyze this {interval} chart for {ticker}.
    
    {market_context}
    
    PROVIDE:
    1. **RECOMMENDATION**: [BUY/SELL/HOLD]
    2. **ANALYSIS**: Key technical and fundamental factors
    3. **RISK ASSESSMENT**: Potential risks and opportunities
    4. **STRATEGY**: Recommended approach based on current conditions
    """
    
    # Generate strategy-specific prompts
    if analysis_type == "Stock Buy/Hold/Sell":
        return _generate_stock_prompt(strategy_type, market_context, interval, ticker)
    elif analysis_type == "Options Trading Strategy":
        return _generate_options_prompt(market_context, interval, ticker)
    else:
        return default_prompt


def _generate_stock_prompt(strategy_type, market_context, interval, ticker):
    """Generate stock-specific prompts."""
    
    if "Short-Term" in strategy_type:
        return f"""
        You are an expert short-term stock trader. Analyze this {interval} chart for {ticker}.
        
        {market_context}
        
        FOCUS ON SHORT-TERM INDICATORS (1-7 days):
        - RSI signals and momentum
        - MACD crossovers and trends
        - Volume patterns and breakouts
        - Short-term moving averages
        - Price action and candlestick patterns
        - Support/resistance levels

        INCORPORATE THE OPTIONS STRATEGY RECOMMENDATION AND RISK MANAGEMENT ABOVE INTO YOUR TRADING PLAN.

        PROVIDE:
        1. **RECOMMENDATION**: [BUY/SELL/HOLD]
        2. **ENTRY POINTS**: Specific price levels
        3. **STOP LOSS**: Based on support levels and ATR
        4. **PROFIT TARGETS**: Multiple take-profit levels
        5. **KEY INDICATORS**: Most relevant signals
        6. **RISK/REWARD**: Ratio and position sizing
        """
    else:  # Long-term
        return f"""
        You are an expert long-term stock analyst. Analyze this {interval} chart for {ticker}.
        
        {market_context}
        
        FOCUS ON LONG-TERM INDICATORS:
        - Trend strength and direction
        - Moving average crossovers
        - Volume trends and accumulation
        - Long-term support/resistance
        - Market sentiment indicators

        INCORPORATE THE OPTIONS STRATEGY RECOMMENDATION AND RISK MANAGEMENT ABOVE INTO YOUR TRADING PLAN.

        PROVIDE:
        1. **RECOMMENDATION**: [BUY/SELL/HOLD]
        2. **TIMEFRAME**: Expected holding period
        3. **ENTRY STRATEGY**: Buy zones and conditions
        4. **RISK MANAGEMENT**: Stop loss levels
        5. **TARGET PRICES**: Based on technical levels
        6. **TREND ANALYSIS**: Primary and secondary trends
        """


def _generate_options_prompt(market_context, interval, ticker):
    """Generate options-specific prompt."""
    
    return f"""
    You are an expert options trader and strategist. Analyze this {interval} chart for {ticker}.
    
    {market_context}
    
    FOCUS ON OPTIONS STRATEGY FACTORS:
    - Implied volatility levels and trends
    - Options flow and unusual activity
    - Time decay considerations
    - Strike selection and expiration timing
    - Risk/reward profiles of different strategies
    - Market volatility expectations

    INCORPORATE THE OPTIONS STRATEGY RECOMMENDATION AND RISK MANAGEMENT ABOVE INTO YOUR TRADING PLAN.

    PROVIDE:
    1. **RECOMMENDATION**: Specific options strategy
    2. **STRIKE SELECTION**: Optimal strike prices and reasoning
    3. **EXPIRATION**: Recommended time to expiration
    4. **ENTRY/EXIT**: Timing and conditions
    5. **RISK MANAGEMENT**: Maximum loss and profit targets
    6. **VOLATILITY PLAY**: How to capitalize on IV changes
    """


def build_market_context(ticker, interval, data, strategy_type, options_strategy, active_indicators, options_strategy_context=""):
    """Build market context string for AI prompts."""
    
    market_context = f"""
    MARKET DATA CONTEXT:
    - Ticker: {ticker}
    - Timeframe: {interval} candles
    - Current Price: ${data['Close'].iloc[-1]:.2f}
    - Strategy Type: {strategy_type}
    - Selected Strategy: {options_strategy}
    - Active Indicators: {', '.join(active_indicators)}
    """

    # Add options strategy context if available
    if options_strategy_context:
        market_context += "\n" + options_strategy_context
    
    return market_context