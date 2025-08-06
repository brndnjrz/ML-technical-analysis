
# =========================
# Imports
# =========================
import plotly.graph_objects as go
import plotly.subplots as sp

# =========================
# Main Chart Generator
# =========================
def create_chart(
    data,
    indicators,
    show_rsi,
    show_macd,
    levels,
    show_adx=False,
    show_stoch=False,
    show_obv=False,
    show_atr=False,
    timeframe="1d"
):
    """
    Generate a multi-row technical analysis chart with overlays and indicators.
    """
    # 1. Define chart rows
    rows_to_plot = [("Candlestick", None)]
    if show_rsi:
        rows_to_plot.append(("RSI", f"RSI_{timeframe}"))
    if show_macd:
        rows_to_plot.append(("MACD", [f"MACD_{timeframe}", f"MACD_Signal_{timeframe}"]))
    if show_adx:
        rows_to_plot.append(("ADX", f'ADX_{timeframe}'))
    if show_stoch:
        rows_to_plot.append(("Stochastic", [f'STOCH_%K_{timeframe}', f'STOCH_%D_{timeframe}']))
    if show_obv:
        rows_to_plot.append(("OBV", f'OBV_{timeframe}'))
    if show_atr:
        rows_to_plot.append(("ATR", f'ATR_{timeframe}'))
    rows_to_plot.append(("Volume", "Volume"))

    # 2. Subplot configuration
    subplot_titles = [title for title, _ in rows_to_plot]
    row_heights = [0.5 if title == "Candlestick" else 0.2 if title == "Volume" else 0.15 for title, _ in rows_to_plot]
    fig = sp.make_subplots(
        rows=len(rows_to_plot),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # 3. Plot each chart type
    for i, (title, data_key) in enumerate(rows_to_plot, start=1):
        if title == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            ), row=i, col=1)
            # Overlays
            for indicator in indicators:
                if indicator == "20-Day SMA" and "SMA_20" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name="SMA 20"), row=i, col=1)
                elif indicator == "50-Day SMA" and "SMA_50" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name="SMA 50"), row=i, col=1)
                elif indicator == "20-Day EMA" and "EMA_20" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name="EMA 20"), row=i, col=1)
                elif indicator == "50-Day EMA" and "EMA_50" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], name="EMA 50"), row=i, col=1)
                elif indicator == "VWAP" and "VWAP" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name="VWAP"), row=i, col=1)
                elif indicator == "Implied Volatility" and "volatility" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['volatility'], name="Implied Volatility"), row=i, col=1)
                elif indicator == "Bollinger Bands":
                    if "BB_upper" in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name="BB Upper"), row=i, col=1)
                    if "BB_middle" in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], name="BB Middle"), row=i, col=1)
                    if "BB_lower" in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name="BB Lower"), row=i, col=1)
            # Support/Resistance
            if levels:
                for s in levels.get("support", []):
                    fig.add_hline(y=s, line_dash="dot", line_color="green", annotation_text="Support", row=i, col=1)
                for r in levels.get("resistance", []):
                    fig.add_hline(y=r, line_dash="dot", line_color="red", annotation_text="Resistance", row=i, col=1)
        elif title == "RSI" and data_key in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[data_key], name="RSI", line=dict(color="blue")), row=i, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
        elif title == "MACD":
            if all(k in data.columns for k in data_key):
                fig.add_trace(go.Scatter(x=data.index, y=data[data_key[0]], name="MACD", line=dict(color="purple")), row=i, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data[data_key[1]], name="Signal", line=dict(color="orange")), row=i, col=1)
        elif title == "ADX" and data_key in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[data_key], name="ADX", line=dict(color="darkviolet")), row=i, col=1)
        elif title == "Stochastic":
            if all(k in data.columns for k in data_key):
                fig.add_trace(go.Scatter(x=data.index, y=data[data_key[0]], name="%K", line=dict(color="blue")), row=i, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data[data_key[1]], name="%D", line=dict(color="green")), row=i, col=1)
        elif title == "OBV" and data_key in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[data_key], name="OBV", line=dict(color="orange")), row=i, col=1)
        elif title == "ATR" and data_key in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[data_key], name="ATR", line=dict(color="gray")), row=i, col=1)
        elif title == "Volume":
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color="lightblue"), row=i, col=1)

    # 4. Final layout
    fig.update_layout(
        height=300 + len(rows_to_plot) * 150,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    return fig

# =========================
# Enhanced Chart Generator
# =========================
def create_enhanced_chart(
    data,
    indicators,
    levels,
    strategy_type=None,
    options_data=None,
    interval="1d"
):
    """
    Enhanced chart function for main app, supporting overlays and strategy-specific features.
    """
    # Convert indicator list to boolean flags
    show_rsi = any('rsi' in str(ind).lower() for ind in indicators)
    show_macd = any('macd' in str(ind).lower() for ind in indicators)
    show_adx = any('adx' in str(ind).lower() for ind in indicators)
    show_stoch = any('stoch' in str(ind).lower() for ind in indicators)
    show_obv = any('obv' in str(ind).lower() for ind in indicators)
    show_atr = any('atr' in str(ind).lower() for ind in indicators)
    # Overlay indicators
    overlay_indicators = []
    for ind in indicators:
        ind_str = str(ind).lower()
        if 'sma' in ind_str:
            if '20' in ind_str:
                overlay_indicators.append("20-Day SMA")
            elif '50' in ind_str:
                overlay_indicators.append("50-Day SMA")
        elif 'ema' in ind_str:
            if '20' in ind_str or '9' in ind_str:
                overlay_indicators.append("20-Day EMA")
            elif '50' in ind_str or '21' in ind_str:
                overlay_indicators.append("50-Day EMA")
        elif 'vwap' in ind_str:
            overlay_indicators.append("VWAP")
        elif 'bollinger' in ind_str or 'bb' in ind_str:
            overlay_indicators.append("Bollinger Bands")
        elif 'volatility' in ind_str or 'iv' in ind_str:
            overlay_indicators.append("Implied Volatility")
    # Use main chart function
    fig = create_chart(
        data=data,
        indicators=overlay_indicators,
        show_rsi=show_rsi,
        show_macd=show_macd,
        levels=levels,
        show_adx=show_adx,
        show_stoch=show_stoch,
        show_obv=show_obv,
        show_atr=show_atr,
        timeframe=interval
    )
    # Strategy-specific enhancements
    if strategy_type and "Short-Term" in strategy_type:
        fig.update_layout(
            title=f"Short-Term Analysis - {interval} Timeframe",
            title_font_size=16
        )
        if len(data) > 20:
            recent_data = data.tail(20)
            fig.add_vrect(
                x0=recent_data.index[0],
                x1=recent_data.index[-1],
                fillcolor="yellow",
                opacity=0.1,
                layer="below",
                line_width=0,
                row=1, col=1
            )
    elif strategy_type and "Long-Term" in strategy_type:
        fig.update_layout(
            title=f"Long-Term Analysis - {interval} Timeframe",
            title_font_size=16
        )
    # Options overlays
    if options_data and options_data.get('iv_data'):
        iv_data = options_data['iv_data']
        fig.add_annotation(
            x=data.index[-1],
            y=data['Close'].iloc[-1],
            text=f"IV Rank: {iv_data.get('iv_rank', 0):.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue",
            bgcolor="lightblue",
            bordercolor="blue",
            borderwidth=1
        )
    # Enhanced layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        hovermode='x unified'
    )
    return fig
