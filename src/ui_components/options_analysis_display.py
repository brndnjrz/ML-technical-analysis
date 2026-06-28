"""Options analysis display component."""
import streamlit as st
from src.analysis.indicators import determine_trend
from src.analysis.options_analysis import analyze_options_chain


def render_options_analysis_section(data, options_data, ticker):
    """Render the options strategy analyzer section."""
    
    st.subheader("Options Strategy Analyzer")
    col1, col2 = st.columns([2, 1])

    with col1:
        # Extract short-term trend only
        short_term_trend = determine_trend(data.tail(20))
        iv_rank = options_data.get('iv_data', {}).get('iv_rank', 50) if options_data else 50
        current_price = data['Close'].iloc[-1]
        
        # Run professional options analysis using our cheatsheet
        options_analysis = analyze_options_chain(
            data,
            ticker,
            current_price,
            short_term_trend,
            iv_rank
        )
        
        # Display options analysis sections
        _render_market_context(short_term_trend, iv_rank, options_analysis)
        _render_strategy_recommendation(options_analysis)
        _render_risk_management(options_analysis)
        
    with col2:
        _render_options_framework(options_analysis)


def _render_market_context(short_term_trend, iv_rank, options_analysis):
    """Render the market context section."""
    st.markdown("### 🔄 Market Context")
    st.markdown(f"- Short-term trend: **{short_term_trend.title()}**")
    st.markdown(f"- IV Rank: **{iv_rank:.1f}%**")

    if 'market_conditions' in options_analysis:
        st.markdown(f"- Market Regime: **{options_analysis['market_conditions']['market_regime']}**")
    else:
        st.warning("Options analysis failed or returned incomplete data. Please check your input or try again.")
        st.code(f"options_analysis keys: {list(options_analysis.keys())}\nFull object: {options_analysis}", language='python')


def _render_strategy_recommendation(options_analysis):
    """Render the strategy recommendation section."""
    st.markdown("### 🎯 Strategy Recommendation")
    strategy = options_analysis.get('strategy_recommendation', {})
    
    expected_keys = ['name', 'trend_direction', 'volatility_environment', 'price_pattern', 'rationale', 'Description']
    missing_keys = [k for k in expected_keys if k not in strategy]
    
    if missing_keys:
        st.warning(f"Strategy recommendation is missing keys: {missing_keys}.\nAvailable keys: {list(strategy.keys()) if strategy else 'None'}\nFull object: {strategy}")
    
    st.markdown(f"**{strategy.get('name', 'N/A')}**")
    st.markdown(f"- **Direction:** {strategy.get('trend_direction', 'N/A').title() if strategy.get('trend_direction') else 'N/A'}")
    st.markdown(f"- **Volatility:** {strategy.get('volatility_environment', 'N/A').title() if strategy.get('volatility_environment') else 'N/A'}")
    st.markdown(f"- **Pattern:** {strategy.get('price_pattern', 'N/A').title() if strategy.get('price_pattern') else 'N/A'}")
    st.markdown(f"- **Rationale:** {strategy.get('rationale', 'N/A')}")
    
    if 'Description' in strategy:
        st.markdown(f"- **Description:** {strategy['Description']}")


def _render_risk_management(options_analysis):
    """Render the risk management section."""
    st.markdown("### ⚠️ Risk Management")
    risk_management = options_analysis.get('risk_management', {})
    rules = risk_management.get('rules', [])
    
    if not rules:
        st.warning(f"No risk management guidance available for this analysis.\nAvailable keys: {list(risk_management.keys()) if risk_management else 'None'}\nFull object: {risk_management}")
    else:
        for rule in rules:
            st.markdown(f"- {rule}")
        st.markdown(f"- Recommended stop: **{risk_management.get('recommended_stop', 'N/A')}**")


def _render_options_framework(options_analysis):
    """Render the professional options framework section."""
    st.markdown("### 📊 Professional Options Framework")
    st.markdown("Our AI uses this 5-step analyst process:")
    
    for step, description in zip(
        options_analysis['analysis_process'].keys(),
        options_analysis['analysis_process'].values()
    ):
        st.markdown(f"**{step.upper()}:** {description}")
    
    st.markdown("---")
    
    # Add option to view the full cheatsheet
    if st.button("📘 View Full Options Strategy Cheatsheet"):
        from src.utils.options_strategy_cheatsheet import get_options_cheatsheet_markdown
        st.markdown(get_options_cheatsheet_markdown())


def get_options_context_for_ai(data, options_data, ticker):
    """
    Generate options context string and variables for AI analysis.
    
    Args:
        data: Stock price data
        options_data: Options chain data  
        ticker: Stock ticker symbol
        
    Returns:
        tuple: (options_strategy_context, options_ai_vars, candidate_strategies, features)
    """
    try:
        short_term_trend = determine_trend(data.tail(20))
        iv_rank = options_data.get('iv_data', {}).get('iv_rank', 50) if options_data else 50
        current_price = data['Close'].iloc[-1]
        
        options_analysis = analyze_options_chain(
            data,
            ticker,
            current_price,
            short_term_trend,
            iv_rank
        )
        
        # Build context string from options_analysis
        strategy = options_analysis.get('strategy_recommendation', {})
        risk_management = options_analysis.get('risk_management', {})
        
        options_strategy_context = f"""
        OPTIONS STRATEGY ANALYZER OUTPUT:
        - Market Regime: {options_analysis.get('market_conditions', {}).get('market_regime', 'N/A')}
        - Recommended Strategy: {strategy.get('name', 'N/A')}
        - Direction: {strategy.get('trend_direction', 'N/A')}
        - Volatility: {strategy.get('volatility_environment', 'N/A')}
        - Pattern: {strategy.get('price_pattern', 'N/A')}
        - Rationale: {strategy.get('rationale', 'N/A')}
        - Description: {strategy.get('Description', 'N/A')}
        - Risk Management: {risk_management.get('rules', [])}
        - Recommended Stop: {risk_management.get('recommended_stop', 'N/A')}"""
        
        options_ai_vars = {
            "options_recommended_stop": risk_management.get('recommended_stop', None),
            "options_direction": strategy.get('trend_direction', None),
            "options_rationale": strategy.get('rationale', None)
        }
        
        # Collect candidate strategy for arbiter
        candidate_strategies = [{
            "name": strategy.get('name'),
            "timeframe": "short_term",  # Default for options
            "trend": strategy.get('trend_direction'),
            "type": strategy.get('name', '').lower().replace(' ', '_'),
            "iv_rank": iv_rank,
            "adx": options_analysis.get('market_conditions', {}).get('adx', 0),
            "rsi": options_analysis.get('market_conditions', {}).get('rsi', 50),
            "confidence": 0.7,  # Placeholder, can be improved
            "rationale": strategy.get('rationale', '')
        }]
        
        features = {
            "trend": strategy.get('trend_direction'),
            "adx": options_analysis.get('market_conditions', {}).get('adx', 0),
            "rsi": options_analysis.get('market_conditions', {}).get('rsi', 50),
            "iv_rank": iv_rank
        }
        
        return options_strategy_context, options_ai_vars, candidate_strategies, features
        
    except Exception as e:
        options_strategy_context = f"OPTIONS STRATEGY ANALYZER OUTPUT: Unavailable due to error: {e}"
        return options_strategy_context, {}, [], {}