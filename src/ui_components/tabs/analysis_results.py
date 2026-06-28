"""Analysis results display component."""
import streamlit as st
import os
from ...utils.formatters import format_analysis_text, format_professional_report
from ...utils.temp_manager import temp_manager
from ...pdf_utils import generate_and_display_pdf


def render_analysis_results(analysis, chart_path, recommendation, ticker, strategy_type, 
                          options_strategy, data, levels, options_data, state_manager):
    """Render the complete AI analysis results section."""
    
    # Display Analysis Results with Enhanced Formatting
    st.markdown("### 🤖 AI Trading Analysis Results")
    st.markdown("---")  # Add a separator line
    
    # Quick Summary Card
    if recommendation:
        _render_summary_card(recommendation)
    
    # Strategy Overview
    if recommendation and 'strategy' in recommendation:
        _render_strategy_overview(recommendation['strategy'])
    
    # Market Analysis Metrics
    if recommendation and 'market_analysis' in recommendation:
        _render_market_conditions(recommendation['market_analysis'])
    
    # Trade Parameters
    if recommendation and 'parameters' in recommendation and recommendation['parameters']:
        _render_trade_parameters(recommendation['parameters'])
    
    # Professional Analysis Display
    _render_professional_report(analysis, recommendation, ticker, strategy_type, 
                              options_strategy, data, levels, options_data)
    
    # Action Buttons
    _render_action_buttons(analysis, recommendation, ticker, strategy_type, 
                         options_strategy, data, levels, options_data, 
                         chart_path, state_manager)


def _render_summary_card(recommendation):
    """Render the analysis summary card."""
    action = recommendation.get('action', 'Hold').upper()
    strategy_name = recommendation.get('strategy', {}).get('name', 'No Strategy')
    confidence = recommendation.get('strategy', {}).get('confidence', 0) * 100
    
    # Color code the action
    if action == 'BUY':
        action_color = "🟢"
    elif action == 'SELL':
        action_color = "🔴"
    else:
        action_color = "🟡"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    ">
        <h4 style="margin: 0; color: #1f77b4;">📋 Analysis Summary</h4>
        <p style="margin: 5px 0; font-size: 18px;">
            <strong>Recommendation:</strong> {action_color} <strong>{action}</strong> 
            | <strong>Strategy:</strong> {strategy_name} 
            | <strong>Confidence:</strong> {confidence:.0f}%
        </p>
    </div>
    """, unsafe_allow_html=True)


def _render_strategy_overview(strategy):
    """Render the strategy overview section."""
    st.markdown("#### 🎯 Strategy Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_name = strategy.get('name', 'N/A')
        st.metric("Strategy", strategy_name, help="Recommended trading strategy")
    
    with col2:
        confidence = strategy.get('confidence', 0) * 100
        confidence_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
        st.metric("Confidence", f"{confidence:.0f}%", help="AI confidence level")
    
    with col3:
        # Get risk_level from recommendation's risk_assessment
        risk_level = strategy.get('risk_level', 'N/A')  # This may need adjustment based on actual data structure
        risk_color = "🔴" if str(risk_level).upper() == "HIGH" else "🟡" if str(risk_level).upper() == "MEDIUM" else "🟢"
        st.metric("Risk Level", f"{risk_color} {str(risk_level).upper()}", help="Assessed risk level")


def _render_market_conditions(market):
    """Render the market conditions section."""
    st.markdown("#### 📊 Current Market Conditions")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        rsi_val = market.get('RSI', 0)
        rsi_signal = "🔴 Overbought" if rsi_val > 70 else "🟢 Oversold" if rsi_val < 30 else "🟡 Neutral"
        st.metric("RSI", f"{rsi_val:.1f}", help=f"Relative Strength Index: {rsi_signal}")
    
    with metrics_col2:
        macd_signal = market.get('MACD_Signal', 'N/A')
        macd_emoji = "🟢" if macd_signal == "bullish" else "🔴" if macd_signal == "bearish" else "🟡"
        st.metric("MACD", f"{macd_emoji} {str(macd_signal).title()}", help="MACD trend signal")
    
    with metrics_col3:
        volume_signal = market.get('volume_signal', 'N/A')
        volume_emoji = "🟢" if volume_signal == "high" else "🔴" if volume_signal == "low" else "🟡"
        st.metric("Volume", f"{volume_emoji} {str(volume_signal).title()}", help="Trading volume analysis")
    
    with metrics_col4:
        trend_strength = market.get('trend_strength', 0)
        trend_signal = "💪 Strong" if trend_strength > 25 else "📈 Weak" if trend_strength > 15 else "➡️ Sideways"
        st.metric("Trend (ADX)", f"{trend_strength:.1f}", help=f"Trend strength: {trend_signal}")


def _render_trade_parameters(params):
    """Render the trade parameters section."""
    st.markdown("#### 📈 Trade Parameters")
    
    # Create formatted parameter display
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        if 'entry_condition' in params and params['entry_condition'] is not None:
            st.info(f"**Entry Condition:** {params['entry_condition'].replace('_', ' ').title()}")
        elif 'entry_condition' in params:
            st.info("**Entry Condition:** Not specified")
        if 'stop_loss' in params:
            try:
                stop_loss_val = float(params['stop_loss'])
                st.error(f"**Stop Loss:** ${stop_loss_val:.2f}")
            except (ValueError, TypeError):
                st.error(f"**Stop Loss:** {params['stop_loss']}")
                
    with param_col2:
        if 'exit_condition' in params and params['exit_condition'] is not None:
            st.info(f"**Exit Condition:** {params['exit_condition'].replace('_', ' ').title()}")
        elif 'exit_condition' in params:
            st.info("**Exit Condition:** Not specified")
        if 'profit_target' in params:
            try:
                profit_val = float(params['profit_target'])
                st.success(f"**Profit Target:** ${profit_val:.2f}")
            except (ValueError, TypeError):
                st.success(f"**Profit Target:** {params['profit_target']}")
                
    # Additional parameters in a clean format
    other_params = {k: v for k, v in params.items() 
                  if k not in ['entry_condition', 'exit_condition', 'stop_loss', 'profit_target']}
    
    if other_params:
        st.markdown("**Additional Parameters:**")
        for key, value in other_params.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, bool):
                value_display = "✅ Yes" if value else "❌ No"
            elif isinstance(value, (int, float)):
                if 'period' in key.lower() or 'ma' in key.lower():
                    value_display = f"{value} periods"
                else:
                    value_display = f"{value:.2f}"
            else:
                if value is not None:
                    value_display = str(value).replace('_', ' ').title()
                else:
                    value_display = "Not specified"
            
            st.write(f"• **{formatted_key}:** {value_display}")


def _render_professional_report(analysis, recommendation, ticker, strategy_type, 
                              options_strategy, data, levels, options_data):
    """Render the professional analysis report."""
    st.markdown("#### 📝 Professional Trade Report")
    
    # Use the new professional formatting function
    try:
        professional_report = format_professional_report(
            analysis, recommendation, ticker, strategy_type, options_strategy, 
            data, levels, options_data
        )
        
        # Display the professional report
        st.markdown(professional_report)
        
    except Exception as format_error:
        st.warning("⚠️ Error formatting professional report. Showing standard format.")
        # Fallback to original formatting
        if analysis:
            cleaned_analysis = format_analysis_text(analysis)
            st.markdown(cleaned_analysis)
        else:
            st.info("📝 No detailed analysis available. Please run the analysis to get AI insights.")


def _render_action_buttons(analysis, recommendation, ticker, strategy_type, 
                         options_strategy, data, levels, options_data, 
                         chart_path, state_manager):
    """Render the action buttons section."""
    st.markdown("---")
    st.markdown("#### 📋 Report & Actions")
    
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        # Enhanced PDF Generation
        if st.button("📄 Generate Detailed Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                try:
                    # Generate professional report for PDF
                    professional_report = format_professional_report(
                        analysis, recommendation, ticker, strategy_type, options_strategy, 
                        data, levels, options_data
                    )
                    
                    generate_and_display_pdf(
                        ticker, strategy_type, options_strategy, data, professional_report, 
                        chart_path, levels, options_data, state_manager.get_active_indicators()
                    )
                    st.success("✅ PDF report generated successfully!")
                    print("✅ PDF report generated successfully")
                    
                    # Clean up temporary chart file after PDF generation
                    if chart_path and os.path.exists(chart_path):
                        temp_manager.cleanup_file(chart_path)
                        print(f"🗑️ Cleaned up temporary chart: {chart_path}")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {e}")
                    print(f"❌ PDF generation error: {e}")
    
    with button_col2:
        # Clean up temp files when user closes analysis
        if st.button("🗑️ Clear Analysis & Clean Temp Files", use_container_width=True):
            state_manager.clear_ai_analysis_result()
            temp_manager.cleanup_all()
            st.success("✅ Analysis cleared and temporary files cleaned up!")
            st.rerun()