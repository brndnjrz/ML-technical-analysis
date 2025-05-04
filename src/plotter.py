import plotly.graph_objects as go
import plotly.subplots as sp

# Generate Plotly Charts
def create_chart(data, indicators, show_rsi, show_macd):
    # Setup Subplots: 
    row_count = 2 + int(show_rsi) + int(show_macd)
    fig = sp.make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5] + ([0.15] if show_rsi else[]) + ([0.15] if show_macd else[]) + [0.2],
        subplot_titles=["Candlestick Chart"] + (["Relative Strength Index (RSI)"] if show_rsi else []) + (["MACD"] if show_macd else []) + ["Volume"]
    )

    current_row = 1

    # Row 1: Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,   # data.index represents the x-axis (dates)
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"  # Replace "trace 0" with "Candlestick")
        ),
        row=current_row,
        col=1
    )

     # Add selected indicators to the chart
    for indicator in indicators:
        # add_indicator(indicator)
        if indicator == "20-Day SMA":
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'), row=current_row, col=1)
        elif indicator == "50-Day SMA":
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
        elif indicator == "20-Day EMA":
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'), row=current_row, col=1)
        elif indicator == "50-Day EMA":
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50'))
        elif indicator == "VWAP":
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'), row=current_row, col=1)
        elif indicator == "Implied Volatility":
            fig.add_trace(go.Scatter(x=data.index, y=data['volatility'], mode='lines', name='Implied Volatility'))
        elif indicator == "Bollinger Bands":
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], mode='lines', name='BB Lower'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_middle'], mode='lines', name='BB Middle'))

    # --- Row 2: RSI (if selected) ---
    if show_rsi:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='blue')),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)

    # --- Row 3: MACD (if selected) ---
    if show_macd:
        current_row += 1
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='purple')),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')),
            row=current_row, col=1
        )

    # --- Last Row: Volume Bars ---
    current_row += 1
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=current_row, col=1
    )

    # Final Chart Formatting and display 
    fig.update_layout(
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False,    # Hides the x-axis range slider
        template="plotly_white"
    )

    return fig



