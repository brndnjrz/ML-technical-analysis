# =========================
# Imports
# =========================
import plotly.graph_objects as go
import plotly.subplots as sp
from src import data_loader
import pandas as pd

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
    timeframe="15m",
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
                # Make sure levels is a dictionary
                if isinstance(levels, dict):
                    # Add support levels
                    if "support" in levels and isinstance(levels["support"], list):
                        for s in levels["support"]:
                            fig.add_hline(y=s, line_dash="dot", line_color="green", annotation_text="Support", row=i, col=1)
                    # Add resistance levels
                    if "resistance" in levels and isinstance(levels["resistance"], list):
                        for r in levels["resistance"]:
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
def get_daily_data(ticker, start_date, end_date):
    """
    Get daily data for a given ticker
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
    
    Returns:
        DataFrame with daily stock data
    """
    try:
        # Fetch daily data using data_loader
        daily_data = data_loader.fetch_stock_data(ticker, start_date, end_date, interval="1d")
        return daily_data
    except Exception as e:
        print(f"Error fetching daily data: {str(e)}")
        # Return None or empty DataFrame on error
        return pd.DataFrame()

def create_enhanced_chart(
    data,
    indicators,
    levels,
    strategy_type=None,
    options_data=None,
    interval="15m"
):
    """
    Enhanced chart function for main app, supporting dual timeframe view:
    - First chart: Daily timeframe for long-term context
    - Second chart: Selected timeframe with indicators
    
    Returns:
        tuple: (subplot_fig, daily_fig, timeframe_fig)
            - subplot_fig: Combined figure with both timeframes for display
            - daily_fig: Separate daily timeframe chart for AI analysis
            - timeframe_fig: Separate selected timeframe chart for AI analysis
    """
    # Extract ticker and date range from the provided data
    ticker = data.attrs.get('ticker', None) if hasattr(data, 'attrs') else None
    
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
    
    # Create a figure with 2 rows - daily chart and selected timeframe chart
    fig = sp.make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=False, 
        vertical_spacing=0.08,
        subplot_titles=(f"Daily Timeframe (Long-Term View)", 
                        f"Selected Timeframe: {interval}")
    )
    
    # Try to get daily data if ticker information is available
    daily_data = None
    try:
        if ticker and not data.empty:
            start_date = data.index.min().date() - pd.Timedelta(days=30)  # Start 30 days earlier for context
            end_date = data.index.max().date()
            daily_data = get_daily_data(ticker, start_date, end_date)
            
            # Make sure we got valid data back
            if daily_data is None or daily_data.empty or len(daily_data) < 5:
                raise ValueError("Insufficient daily data points")
    except Exception as e:
        print(f"Error getting daily data: {str(e)}")
        daily_data = None
    
    # If we couldn't get daily data, use the provided data for both charts
    if daily_data is None or daily_data.empty:
        daily_data = data
    
    # 1. Add daily candlestick chart (top)
    fig.add_trace(go.Candlestick(
        x=daily_data.index,
        open=daily_data['Open'],
        high=daily_data['High'],
        low=daily_data['Low'],
        close=daily_data['Close'],
        name="Daily"
    ), row=1, col=1)
    
    # Add simple moving averages to daily chart for context
    if 'SMA_50' in daily_data.columns:
        fig.add_trace(go.Scatter(
            x=daily_data.index, 
            y=daily_data['SMA_50'], 
            name="50 SMA (Daily)", 
            line=dict(color='blue')
        ), row=1, col=1)
    
    if 'SMA_200' in daily_data.columns:
        fig.add_trace(go.Scatter(
            x=daily_data.index, 
            y=daily_data['SMA_200'], 
            name="200 SMA (Daily)", 
            line=dict(color='red')
        ), row=1, col=1)
    
    # Add support/resistance to daily chart
    if isinstance(levels, dict):
        if "support" in levels and isinstance(levels["support"], list):
            for s in levels["support"]:
                fig.add_hline(y=s, line_dash="dot", line_color="green", 
                              annotation_text="Support", row=1, col=1)
        if "resistance" in levels and isinstance(levels["resistance"], list):
            for r in levels["resistance"]:
                fig.add_hline(y=r, line_dash="dot", line_color="red", 
                              annotation_text="Resistance", row=1, col=1)
    
    # 2. Add selected timeframe chart with indicators (bottom)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f"{interval}"
    ), row=2, col=1)
    
    # Add indicators to selected timeframe chart
    for ind in overlay_indicators:
        if ind == "20-Day SMA":
            col = f"SMA_20_{interval}" if f"SMA_20_{interval}" in data.columns else "SMA_20"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name="SMA 20", line=dict(color='purple')
                ), row=2, col=1)
        elif ind == "50-Day SMA":
            col = f"SMA_50_{interval}" if f"SMA_50_{interval}" in data.columns else "SMA_50"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name="SMA 50", line=dict(color='blue')
                ), row=2, col=1)
        elif ind == "20-Day EMA":
            col = f"EMA_20_{interval}" if f"EMA_20_{interval}" in data.columns else "EMA_20"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name="EMA 20", line=dict(color='orange')
                ), row=2, col=1)
        elif ind == "50-Day EMA":
            col = f"EMA_50_{interval}" if f"EMA_50_{interval}" in data.columns else "EMA_50"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name="EMA 50", line=dict(color='green')
                ), row=2, col=1)
        elif ind == "VWAP":
            col = f"VWAP_{interval}" if f"VWAP_{interval}" in data.columns else "VWAP"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name="VWAP", line=dict(color='purple')
                ), row=2, col=1)
        elif ind == "Bollinger Bands":
            upper = f"BB_upper_{interval}" if f"BB_upper_{interval}" in data.columns else "BB_upper"
            middle = f"BB_middle_{interval}" if f"BB_middle_{interval}" in data.columns else "BB_middle"
            lower = f"BB_lower_{interval}" if f"BB_lower_{interval}" in data.columns else "BB_lower"
            if upper in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[upper], name="BB Upper", line=dict(color='rgba(0,0,255,0.5)')
                ), row=2, col=1)
            if middle in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[middle], name="BB Middle", line=dict(color='rgba(0,0,255,0.3)')
                ), row=2, col=1)
            if lower in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[lower], name="BB Lower", line=dict(color='rgba(0,0,255,0.5)')
                ), row=2, col=1)
    
    # Add support/resistance to selected timeframe chart
    if isinstance(levels, dict):
        if "support" in levels and isinstance(levels["support"], list):
            for s in levels["support"]:
                fig.add_hline(y=s, line_dash="dot", line_color="green", 
                              annotation_text="Support", row=2, col=1)
        if "resistance" in levels and isinstance(levels["resistance"], list):
            for r in levels["resistance"]:
                fig.add_hline(y=r, line_dash="dot", line_color="red", 
                              annotation_text="Resistance", row=2, col=1)
    # Add secondary indicators below the main chart
    # Only add these if we have enough space in the figure
    if show_rsi or show_macd or show_adx or show_stoch:
        # Add indicators in separate row - create another subplot
        indicator_fig = go.Figure()
        
        if show_rsi:
            col = f"RSI_{interval}" if f"RSI_{interval}" in data.columns else "RSI"
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name="RSI", line=dict(color="blue")
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
        if show_macd:
            macd_col = f"MACD_{interval}" if f"MACD_{interval}" in data.columns else "MACD"
            signal_col = f"MACD_Signal_{interval}" if f"MACD_Signal_{interval}" in data.columns else "MACD_Signal"
            if macd_col in data.columns and signal_col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[macd_col], name="MACD", line=dict(color="purple")
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[signal_col], name="Signal", line=dict(color="orange")
                ), row=2, col=1)

    # Strategy-specific enhancements
    if strategy_type and "Short-Term" in strategy_type:
        fig.update_layout(
            title=f"Technical Analysis: Daily and {interval} Timeframes",
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
                row=2, col=1  # Highlight in the selected timeframe chart
            )
    elif strategy_type and "Long-Term" in strategy_type:
        fig.update_layout(
            title=f"Technical Analysis: Daily and {interval} Timeframes",
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
            borderwidth=1,
            row=2, col=1  # Add to the selected timeframe chart
        )
    
    # Enhanced layout
    fig.update_layout(
        height=1000,  # Increased height for dual chart
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        template="plotly_white",
        hovermode='x unified'
    )
    
    # Update y-axis ranges with padding
    if not data.empty:
        # For daily chart
        daily_min = daily_data['Low'].min() if not daily_data.empty else data['Low'].min()
        daily_max = daily_data['High'].max() if not daily_data.empty else data['High'].max()
        daily_padding = (daily_max - daily_min) * 0.05
        fig.update_yaxes(range=[daily_min - daily_padding, daily_max + daily_padding], row=1, col=1)
        
        # For selected timeframe chart
        min_price = data['Low'].min()
        max_price = data['High'].max()
        price_padding = (max_price - min_price) * 0.05
        fig.update_yaxes(range=[min_price - price_padding, max_price + price_padding], row=2, col=1)
    
    # Create separate figures for AI analysis
    # 1. Daily Chart
    daily_fig = go.Figure()
    
    # Add daily candlesticks
    if not daily_data.empty:
        daily_fig.add_trace(go.Candlestick(
            x=daily_data.index,
            open=daily_data['Open'],
            high=daily_data['High'],
            low=daily_data['Low'],
            close=daily_data['Close'],
            name=f"{ticker} Daily",
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        # Add support and resistance to daily chart
        if "support" in levels and isinstance(levels["support"], list):
            for s in levels["support"]:
                daily_fig.add_hline(y=s, line_dash="dot", line_color="green", 
                                   annotation_text="Support")
        if "resistance" in levels and isinstance(levels["resistance"], list):
            for r in levels["resistance"]:
                daily_fig.add_hline(y=r, line_dash="dot", line_color="red", 
                                   annotation_text="Resistance")
    
    daily_fig.update_layout(
        title=f"{ticker} - Daily Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600
    )
    
    # 2. Timeframe Chart (selected interval)
    timeframe_fig = go.Figure()
    
    # Add selected timeframe candlesticks
    timeframe_fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f"{ticker} {interval}",
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add support and resistance to timeframe chart
    if "support" in levels and isinstance(levels["support"], list):
        for s in levels["support"]:
            timeframe_fig.add_hline(y=s, line_dash="dot", line_color="green", 
                                   annotation_text="Support")
    if "resistance" in levels and isinstance(levels["resistance"], list):
        for r in levels["resistance"]:
            timeframe_fig.add_hline(y=r, line_dash="dot", line_color="red", 
                                   annotation_text="Resistance")
    
    # Add the primary indicators to the timeframe chart
    for indicator in indicators:
        if indicator in data.columns:
            timeframe_fig.add_trace(go.Scatter(
                x=data.index,
                y=data[indicator],
                name=indicator,
                line=dict(width=1)
            ))
    
    timeframe_fig.update_layout(
        title=f"{ticker} - {interval} Chart",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600
    )
    
    return fig, daily_fig, timeframe_fig
