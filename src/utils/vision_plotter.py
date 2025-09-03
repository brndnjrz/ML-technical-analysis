"""
Enhanced deterministic chart generation for AI vision analysis.
Creates standardized, labeled charts optimized for machine learning models.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import io
import base64
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Fixed theme for deterministic visuals
VISION_THEME = {
    'background': '#FFFFFF',
    'grid': '#E8E8E8',
    'text': '#000000',
    'candlestick': {
        'increasing': '#00BF63',  # Green
        'decreasing': '#FF4B4B',  # Red
    },
    'indicators': {
        'ma_20': '#FF6B6B',      # Light red
        'ma_50': '#4ECDC4',      # Teal
        'ma_200': '#45B7D1',     # Blue
        'vwap': '#FFA726',       # Orange
        'bollinger': '#9C27B0',  # Purple
        'support': '#4CAF50',    # Green
        'resistance': '#F44336', # Red
    },
    'rsi': {
        'line': '#2196F3',
        'overbought': '#FF5722',
        'oversold': '#4CAF50',
    }
}

def create_vision_optimized_chart(
    data: pd.DataFrame,
    ticker: str,
    timeframe: str = "1d",
    current_price: float = None,
    support_levels: List[float] = None,
    resistance_levels: List[float] = None,
    width: int = 800,
    height: int = 800,
    show_annotations: bool = True
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Create deterministic chart optimized for AI vision analysis
    
    Returns:
        Tuple of (figure, metadata_dict)
    """
    try:
        if data.empty or len(data) < 20:
            raise ValueError("Insufficient data for chart generation")
        
        # Initialize support/resistance if not provided
        support_levels = support_levels or []
        resistance_levels = resistance_levels or []
        current_price = current_price or data['Close'].iloc[-1]
        
        # Create subplot with fixed layout
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                f"{ticker} - {timeframe.upper()} PRICE ACTION",
                "RSI (14)",
                "VOLUME"
            ],
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Main price chart with candlesticks
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color=VISION_THEME['candlestick']['increasing'],
            decreasing_line_color=VISION_THEME['candlestick']['decreasing'],
            name="Price",
            showlegend=False
        ), row=1, col=1)
        
        # 2. Add key moving averages with distinct styles
        if 'SMA_20' in data.columns or 'EMA_20' in data.columns:
            ma_20_col = 'SMA_20' if 'SMA_20' in data.columns else 'EMA_20'
            if not data[ma_20_col].isna().all():
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[ma_20_col],
                    name="20 MA",
                    line=dict(color=VISION_THEME['indicators']['ma_20'], width=2),
                    showlegend=False
                ), row=1, col=1)
        
        if 'SMA_50' in data.columns or 'EMA_50' in data.columns:
            ma_50_col = 'SMA_50' if 'SMA_50' in data.columns else 'EMA_50'
            if not data[ma_50_col].isna().all():
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[ma_50_col],
                    name="50 MA",
                    line=dict(color=VISION_THEME['indicators']['ma_50'], width=2),
                    showlegend=False
                ), row=1, col=1)
        
        if 'SMA_200' in data.columns:
            if not data['SMA_200'].isna().all():
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_200'],
                    name="200 MA",
                    line=dict(color=VISION_THEME['indicators']['ma_200'], width=3),
                    showlegend=False
                ), row=1, col=1)
        
        # 3. VWAP if available
        if 'VWAP' in data.columns and not data['VWAP'].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['VWAP'],
                name="VWAP",
                line=dict(color=VISION_THEME['indicators']['vwap'], width=2, dash='dash'),
                showlegend=False
            ), row=1, col=1)
        
        # 4. Bollinger Bands if available
        if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
            if not (data['BB_upper'].isna().all() or data['BB_lower'].isna().all()):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_upper'],
                    name="BB Upper",
                    line=dict(color=VISION_THEME['indicators']['bollinger'], width=1),
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_lower'],
                    name="BB Lower",
                    line=dict(color=VISION_THEME['indicators']['bollinger'], width=1),
                    fill='tonexty',
                    fillcolor='rgba(156, 39, 176, 0.1)',
                    showlegend=False
                ), row=1, col=1)
        
        # 5. Support and Resistance levels
        x_range = [data.index.min(), data.index.max()]
        
        for level in support_levels:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[level, level],
                name=f"Support ${level:.2f}",
                line=dict(color=VISION_THEME['indicators']['support'], width=2, dash='solid'),
                showlegend=False
            ), row=1, col=1)
        
        for level in resistance_levels:
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[level, level],
                name=f"Resistance ${level:.2f}",
                line=dict(color=VISION_THEME['indicators']['resistance'], width=2, dash='solid'),
                showlegend=False
            ), row=1, col=1)
        
        # 6. Current price line
        fig.add_trace(go.Scatter(
            x=x_range,
            y=[current_price, current_price],
            name=f"Current ${current_price:.2f}",
            line=dict(color='#333333', width=3, dash='dot'),
            showlegend=False
        ), row=1, col=1)
        
        # 7. RSI subplot
        if 'RSI' in data.columns and not data['RSI'].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name="RSI",
                line=dict(color=VISION_THEME['rsi']['line'], width=2),
                showlegend=False
            ), row=2, col=1)
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color=VISION_THEME['rsi']['overbought'], 
                         annotation_text="Overbought (70)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=VISION_THEME['rsi']['oversold'], 
                         annotation_text="Oversold (30)", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="#888888", 
                         annotation_text="Neutral (50)", row=2, col=1)
        
        # 8. Volume subplot
        if 'Volume' in data.columns and not data['Volume'].isna().all():
            colors = ['green' if row['Close'] > row['Open'] else 'red' 
                     for _, row in data.iterrows()]
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color=colors,
                showlegend=False,
                opacity=0.7
            ), row=3, col=1)
        
        # 9. Layout configuration for vision analysis
        fig.update_layout(
            title={
                'text': f"{ticker} Technical Analysis - {timeframe}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': VISION_THEME['text']}
            },
            width=width,
            height=height,
            paper_bgcolor=VISION_THEME['background'],
            plot_bgcolor=VISION_THEME['background'],
            font={'color': VISION_THEME['text'], 'size': 12},
            showlegend=False,  # Clean appearance for vision analysis
            margin=dict(l=60, r=60, t=60, b=60),
            xaxis_rangeslider_visible=False
        )
        
        # Update all axes
        fig.update_xaxes(
            showgrid=True,
            gridcolor=VISION_THEME['grid'],
            gridwidth=1,
            showline=True,
            linecolor=VISION_THEME['text'],
            linewidth=1
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=VISION_THEME['grid'],
            gridwidth=1,
            showline=True,
            linecolor=VISION_THEME['text'],
            linewidth=1
        )
        
        # Specific formatting for price axis
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        # 10. Add contextual annotations if enabled
        if show_annotations:
            _add_context_annotations(fig, data, current_price, support_levels, resistance_levels)
        
        # 11. Generate metadata
        metadata = _generate_chart_metadata(data, ticker, timeframe, current_price, 
                                          support_levels, resistance_levels)
        
        logger.info(f"✅ Vision-optimized chart generated: {ticker} {timeframe} "
                   f"({len(data)} candles, {len(support_levels)} support, {len(resistance_levels)} resistance)")
        
        return fig, metadata
        
    except Exception as e:
        logger.error(f"❌ Chart generation error: {e}")
        # Return minimal fallback chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0], y=[0], text=["Chart generation failed"]))
        fig.update_layout(title="Chart Error", width=width, height=height)
        return fig, {'error': str(e)}

def _add_context_annotations(fig: go.Figure, data: pd.DataFrame, current_price: float,
                           support_levels: List[float], resistance_levels: List[float]):
    """Add contextual annotations to help vision analysis"""
    try:
        # Current price annotation
        fig.add_annotation(
            x=data.index[-1],
            y=current_price,
            text=f"Current: ${current_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#333333",
            bgcolor="white",
            bordercolor="#333333",
            font=dict(size=12, color="#333333")
        )
        
        # RSI current value if available
        if 'RSI' in data.columns and not data['RSI'].isna().all():
            current_rsi = data['RSI'].iloc[-1]
            rsi_condition = "OVERBOUGHT" if current_rsi > 70 else "OVERSOLD" if current_rsi < 30 else "NEUTRAL"
            
            fig.add_annotation(
                x=data.index[-1],
                y=current_rsi,
                text=f"RSI: {current_rsi:.1f} ({rsi_condition})",
                showarrow=False,
                bgcolor="white",
                bordercolor="#2196F3",
                font=dict(size=10, color="#2196F3"),
                row=2, col=1
            )
        
        # Trend indication based on moving averages
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            ma20 = data['SMA_20'].iloc[-1]
            ma50 = data['SMA_50'].iloc[-1]
            
            if not (pd.isna(ma20) or pd.isna(ma50)):
                trend_text = "BULLISH TREND" if ma20 > ma50 else "BEARISH TREND"
                trend_color = "#00BF63" if ma20 > ma50 else "#FF4B4B"
                
                fig.add_annotation(
                    x=data.index[len(data)//2],  # Middle of chart
                    y=max(data['High'].iloc[-20:]),  # Top area
                    text=trend_text,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor=trend_color,
                    font=dict(size=14, color=trend_color, family="Arial Black")
                )
        
    except Exception as e:
        logger.warning(f"⚠️ Annotation error: {e}")

def _generate_chart_metadata(data: pd.DataFrame, ticker: str, timeframe: str, 
                           current_price: float, support_levels: List[float], 
                           resistance_levels: List[float]) -> Dict[str, Any]:
    """Generate comprehensive metadata for the chart"""
    try:
        # Basic stats
        metadata = {
            'ticker': ticker,
            'timeframe': timeframe,
            'candles_count': len(data),
            'date_range': {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            },
            'current_price': current_price,
            'price_range': {
                'high_52w': float(data['High'].max()),
                'low_52w': float(data['Low'].min())
            },
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
        }
        
        # Technical indicators summary
        if 'RSI' in data.columns and not data['RSI'].isna().all():
            current_rsi = data['RSI'].iloc[-1]
            metadata['rsi'] = {
                'current': float(current_rsi),
                'condition': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral'
            }
        
        if 'ATR' in data.columns and not data['ATR'].isna().all():
            current_atr = data['ATR'].iloc[-1]
            metadata['atr'] = {
                'current': float(current_atr),
                'percentage': float((current_atr / current_price) * 100)
            }
        
        # Moving average positions
        ma_data = {}
        for period in [20, 50, 200]:
            col_sma = f'SMA_{period}'
            col_ema = f'EMA_{period}'
            
            if col_sma in data.columns and not data[col_sma].isna().all():
                ma_value = data[col_sma].iloc[-1]
                ma_data[f'ma_{period}'] = {
                    'value': float(ma_value),
                    'position': 'above' if current_price > ma_value else 'below'
                }
            elif col_ema in data.columns and not data[col_ema].isna().all():
                ma_value = data[col_ema].iloc[-1]
                ma_data[f'ma_{period}'] = {
                    'value': float(ma_value),
                    'position': 'above' if current_price > ma_value else 'below'
                }
        
        metadata['moving_averages'] = ma_data
        
        # Volume analysis
        if 'Volume' in data.columns and not data['Volume'].isna().all():
            recent_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            metadata['volume'] = {
                'current': int(recent_volume),
                'avg_20d': int(avg_volume),
                'relative': 'high' if recent_volume > avg_volume * 1.5 else 'low' if recent_volume < avg_volume * 0.5 else 'normal'
            }
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Metadata generation error: {e}")
        return {'ticker': ticker, 'error': str(e)}

def export_chart_for_vision(fig: go.Figure, format: str = 'webp', 
                          max_size_kb: int = 250, quality: int = 85) -> bytes:
    """
    Export chart optimized for vision analysis
    
    Args:
        fig: Plotly figure
        format: 'webp', 'png', or 'jpeg'  
        max_size_kb: Maximum file size in KB
        quality: Compression quality (1-100)
    
    Returns:
        bytes: Compressed image data
    """
    try:
        # Export as PNG first (highest quality)
        img_bytes = fig.to_image(format="png", width=800, height=800, scale=1)
        
        # Convert to PIL Image for optimization
        img = Image.open(io.BytesIO(img_bytes))
        
        # Optimize size and format
        output = io.BytesIO()
        
        if format.lower() == 'webp':
            # WebP provides best compression
            img.save(output, format='WEBP', quality=quality, optimize=True)
        elif format.lower() == 'jpeg':
            # JPEG for photos, but PNG is better for charts
            if img.mode in ('RGBA', 'LA'):
                # Convert to RGB for JPEG
                background = Image.new('RGB', img.size, 'white')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            img.save(output, format='JPEG', quality=quality, optimize=True)
        else:
            # PNG with optimization
            img.save(output, format='PNG', optimize=True)
        
        compressed_bytes = output.getvalue()
        
        # Check size and adjust quality if needed
        size_kb = len(compressed_bytes) / 1024
        if size_kb > max_size_kb and quality > 30:
            # Recursively reduce quality
            return export_chart_for_vision(fig, format, max_size_kb, quality - 15)
        
        logger.info(f"✅ Chart exported: {size_kb:.1f}KB {format.upper()}")
        return compressed_bytes
        
    except Exception as e:
        logger.error(f"❌ Chart export error: {e}")
        # Return minimal fallback
        return b"Chart export failed"

def create_chart_with_watermark(data: pd.DataFrame, ticker: str, metadata: Dict[str, Any]) -> go.Figure:
    """Add informational watermark to chart for vision model context"""
    fig, _ = create_vision_optimized_chart(data, ticker)
    
    try:
        # Add watermark with key context
        watermark_text = (
            f"{ticker} | {metadata.get('timeframe', 'Unknown')} | "
            f"Price: ${metadata.get('current_price', 0):.2f} | "
            f"RSI: {metadata.get('rsi', dict()).get('current', 50):.0f} | "
            f"ATR: {metadata.get('atr', dict()).get('percentage', 2):.1f}%"
        )
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=watermark_text,
            showarrow=False,
            font=dict(size=10, color="#666666"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#CCCCCC",
            borderwidth=1
        )
        
    except Exception as e:
        logger.warning(f"⚠️ Watermark error: {e}")
    
    return fig
