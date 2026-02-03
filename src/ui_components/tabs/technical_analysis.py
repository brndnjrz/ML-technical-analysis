"""Technical Analysis tab component."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ...utils.app_config import INDICATOR_GROUPS
from ... import plotter


def render_technical_analysis_tab(data, levels, options_data, active_indicators, interval, fundamentals):
    """Render the technical analysis tab content."""
    
    # Show key metrics at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price",
            f"${data['Close'].iloc[-1]:.2f}",
            f"{((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100:.2f}%"
        )

    with col2:
        if levels['support']:
            nearest_support = max([s for s in levels['support'] if s < data['Close'].iloc[-1]], default=0)
            st.metric("Nearest Support", f"${nearest_support:.2f}")

    with col3:
        if levels['resistance']:
            nearest_resistance = min([r for r in levels['resistance'] if r > data['Close'].iloc[-1]], default=0)
            st.metric("Nearest Resistance", f"${nearest_resistance:.2f}")

    # Stock Metrics Section
    _render_stock_metrics(options_data, fundamentals)
    
    # Technical Analysis Chart
    st.markdown("### 📈 Technical Analysis Chart")
    
    # Current Indicator Values
    _render_current_indicators(data)
    
    # Create and display chart
    _render_chart(data, active_indicators, levels, interval)
    
    # Technical Indicators Summary
    _render_indicator_summary(data)


def _render_stock_metrics(options_data, fundamentals):
    """Render the stock metrics section."""
    st.markdown("### 📊 Stock Metrics")
    metric_labels = [
        ("IV Rank", f"{options_data['iv_data'].get('iv_rank', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("IV Percentile", f"{options_data['iv_data'].get('iv_percentile', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("30-Day HV", f"{options_data['iv_data'].get('hv_30', 0):.1f}%" if options_data and options_data.get("iv_data") else "N/A"),
        ("VIX Level", f"{options_data['iv_data'].get('vix', 0):.1f}" if options_data and options_data.get("iv_data") else "N/A"),
        ("EPS", f"{fundamentals.get('EPS', 0):.2f}" if isinstance(fundamentals.get('EPS'), (int, float)) else "N/A"),
        ("P/E Ratio", f"{fundamentals.get('P/E Ratio', 0):.2f}" if isinstance(fundamentals.get('P/E Ratio'), (int, float)) else "N/A"),
        ("Revenue Growth", f"{fundamentals.get('Revenue Growth', 0):.1f}%" if isinstance(fundamentals.get('Revenue Growth'), (int, float)) else "N/A"),
        ("Profit Margin", f"{fundamentals.get('Profit Margin', 0):.1f}%" if isinstance(fundamentals.get('Profit Margin'), (int, float)) else "N/A"),
    ]

    cols = st.columns(4)
    for i, (label, value) in enumerate(metric_labels):
        with cols[i % 4]:
            st.metric(label, value if value is not None else "N/A")


def _render_current_indicators(data):
    """Render current indicator values in an expandable section."""
    with st.expander("📊 Current Indicator Values", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        latest = data.iloc[-1]
        
        with col1:
            st.markdown("**Trend Indicators**")
            if 'SMA_20' in data.columns and pd.notna(latest['SMA_20']):
                st.metric("SMA(20)", f"${latest['SMA_20']:.2f}")
            if 'EMA_20' in data.columns and pd.notna(latest['EMA_20']):
                st.metric("EMA(20)", f"${latest['EMA_20']:.2f}")
            if 'VWAP' in data.columns and pd.notna(latest['VWAP']):
                st.metric("VWAP", f"${latest['VWAP']:.2f}")
        
        with col2:
            st.markdown("**Momentum Indicators**")
            if 'RSI' in data.columns and pd.notna(latest['RSI']):
                rsi_color = "🔴" if latest['RSI'] > 70 else "🟢" if latest['RSI'] < 30 else "🟡"
                st.metric("RSI(14)", f"{latest['RSI']:.1f} {rsi_color}")
            if 'MACD' in data.columns and pd.notna(latest['MACD']):
                st.metric("MACD", f"{latest['MACD']:.4f}")
            if 'ADX' in data.columns and pd.notna(latest['ADX']):
                st.metric("ADX(14)", f"{latest['ADX']:.1f}")
        
        with col3:
            st.markdown("**Volatility & Volume**")
            if 'ATR' in data.columns and pd.notna(latest['ATR']):
                st.metric("ATR(14)", f"${latest['ATR']:.2f}")
            if 'OBV' in data.columns and pd.notna(latest['OBV']):
                st.metric("OBV", f"{latest['OBV']:,.0f}")
            if 'volatility' in data.columns and pd.notna(latest['volatility']):
                st.metric("Historical Vol", f"{latest['volatility']*100:.1f}%")


def _render_chart(data, active_indicators, levels, interval):
    """Render the main technical analysis chart."""
    try:
        # Ensure levels is a properly formatted dictionary
        if not isinstance(levels, dict) or not ("support" in levels and "resistance" in levels):
            levels = {'support': [], 'resistance': []}
            
        subplot_fig, daily_fig, timeframe_fig = plotter.create_enhanced_chart(
            data=data,
            indicators=active_indicators, 
            levels=levels,
            interval=interval
        )
    except Exception as chart_error:
        st.error(f"Error creating chart: {str(chart_error)}")
        # Create a simple fallback chart
        subplot_fig = go.Figure()
        daily_fig = go.Figure() 
        timeframe_fig = go.Figure()
        # Add data to all charts
        for chart in [subplot_fig, daily_fig, timeframe_fig]:
            chart.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
    
    st.plotly_chart(subplot_fig, use_container_width=True)


def _render_indicator_summary(data):
    """Render technical indicators summary."""
    st.markdown("### Technical Indicators Summary")
    
    # Create columns for indicator categories
    ind_cols = st.columns(len(INDICATOR_GROUPS))
    
    # Display indicator values if available
    for i, (group_name, indicators) in enumerate(INDICATOR_GROUPS.items()):
        with ind_cols[i]:
            st.markdown(f"**{group_name}**")
            for ind in indicators:
                for col in data.columns:
                    if ind in col:
                        try:
                            value = data[col].iloc[-1]
                            if isinstance(value, (int, float)):
                                st.metric(col, f"{value:.2f}")
                        except:
                            pass