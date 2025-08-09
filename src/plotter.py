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
    timeframe="1d",
    yaxis_range=None,
    xaxis_range=None
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
                if indicator == "20-Day SMA":
                    col = f"SMA_20_{timeframe}" if f"SMA_20_{timeframe}" in data.columns else "SMA_20"
                    if col in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[col], name="SMA 20"), row=i, col=1)
                elif indicator == "50-Day SMA":
                    col = f"SMA_50_{timeframe}" if f"SMA_50_{timeframe}" in data.columns else "SMA_50"
                    if col in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[col], name="SMA 50"), row=i, col=1)
                elif indicator == "20-Day EMA":
                    col = f"EMA_20_{timeframe}" if f"EMA_20_{timeframe}" in data.columns else "EMA_20"
                    if col in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[col], name="EMA 20"), row=i, col=1)
                elif indicator == "50-Day EMA":
                    col = f"EMA_50_{timeframe}" if f"EMA_50_{timeframe}" in data.columns else "EMA_50"
                    if col in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[col], name="EMA 50"), row=i, col=1)
                elif indicator == "VWAP":
                    col = f"VWAP_{timeframe}" if f"VWAP_{timeframe}" in data.columns else "VWAP"
                    if col in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[col], name="VWAP"), row=i, col=1)
                elif indicator == "Implied Volatility" and "volatility" in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['volatility'], name="Implied Volatility"), row=i, col=1)
                elif indicator == "Bollinger Bands":
                    upper = f"BB_upper_{timeframe}" if f"BB_upper_{timeframe}" in data.columns else "BB_upper"
                    middle = f"BB_middle_{timeframe}" if f"BB_middle_{timeframe}" in data.columns else "BB_middle"
                    lower = f"BB_lower_{timeframe}" if f"BB_lower_{timeframe}" in data.columns else "BB_lower"
                    if upper in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[upper], name="BB Upper"), row=i, col=1)
                    if middle in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[middle], name="BB Middle"), row=i, col=1)
                    if lower in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data[lower], name="BB Lower"), row=i, col=1)
            # Support/Resistance
            if levels:
                for s in levels.get("support", []):
                    fig.add_hline(y=s, line_dash="dot", line_color="green", annotation_text="Support", row=i, col=1)
                for r in levels.get("resistance", []):
                    fig.add_hline(y=r, line_dash="dot", line_color="red", annotation_text="Resistance", row=i, col=1)
        elif title == "RSI":
            col = f"RSI_{timeframe}" if f"RSI_{timeframe}" in data.columns else "RSI"
            if col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name="RSI", line=dict(color="blue")), row=i, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
        elif title == "MACD":
            macd_col = f"MACD_{timeframe}" if f"MACD_{timeframe}" in data.columns else "MACD"
            signal_col = f"MACD_Signal_{timeframe}" if f"MACD_Signal_{timeframe}" in data.columns else "MACD_Signal"
            if macd_col in data.columns and signal_col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[macd_col], name="MACD", line=dict(color="purple")), row=i, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data[signal_col], name="Signal", line=dict(color="orange")), row=i, col=1)
        elif title == "ADX":
            col = f"ADX_{timeframe}" if f"ADX_{timeframe}" in data.columns else "ADX"
            if col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name="ADX", line=dict(color="darkviolet")), row=i, col=1)
        elif title == "Stochastic":
            k_col = f"STOCH_%K_{timeframe}" if f"STOCH_%K_{timeframe}" in data.columns else "STOCH_%K"
            d_col = f"STOCH_%D_{timeframe}" if f"STOCH_%D_{timeframe}" in data.columns else "STOCH_%D"
            if k_col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[k_col], name="%K", line=dict(color="blue")), row=i, col=1)
            if d_col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[d_col], name="%D", line=dict(color="green")), row=i, col=1)
        elif title == "OBV":
            col = f"OBV_{timeframe}" if f"OBV_{timeframe}" in data.columns else "OBV"
            if col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name="OBV", line=dict(color="orange")), row=i, col=1)
        elif title == "ATR":
            col = f"ATR_{timeframe}" if f"ATR_{timeframe}" in data.columns else "ATR"
            if col in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col], name="ATR", line=dict(color="gray")), row=i, col=1)
        elif title == "Volume":
            if "Volume" in data.columns:
                fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color="lightblue"), row=i, col=1)

    # Calculate a padded y-axis range if not provided
    if yaxis_range is None:
        min_price = data['Low'].min()
        max_price = data['High'].max()
        price_padding = (max_price - min_price) * 0.05  # 5% padding
        yaxis_range = [min_price - price_padding, max_price + price_padding]

    # Show the full chart for the selected timeframe
    if xaxis_range is None:
        xaxis_range = [data.index.min(), data.index.max()]

    # 4. Final layout
    fig.update_layout(
        height=300 + len(rows_to_plot) * 150,
        showlegend=True,
        xaxis_rangeslider_visible=True,
        template="plotly_white"
    )
    fig.update_yaxes(range=yaxis_range, row=1, col=1)
    fig.update_xaxes(range=xaxis_range, row=1, col=1)
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
        timeframe=interval,
        # yaxis_range and xaxis_range are handled automatically
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
