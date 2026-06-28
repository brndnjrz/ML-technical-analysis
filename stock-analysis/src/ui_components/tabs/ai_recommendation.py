"""AI Recommendation tab component."""
import streamlit as st
from ...utils.formatters import format_analysis_text


def render_ai_recommendation_tab(state_manager):
    """Render the AI recommendation tab content."""
    st.markdown("### 🤖 AI-Powered Strategy Analysis")
    
    # Check if we already have analysis results
    ai_result = state_manager.get_ai_analysis_result()
    
    if ai_result:
        analysis, chart_path, recommendation = ai_result
        _render_analysis_results(analysis, recommendation, state_manager)
    else:
        # No analysis yet, show the Run Analysis button
        _render_run_analysis_button(state_manager)


def _render_run_analysis_button(state_manager):
    """Render the run analysis button."""
    analysis_cols = st.columns([1])
    
    with analysis_cols[0]:
        run_analysis = st.button("Run Analysis 💸", type="primary", use_container_width=True)
        if run_analysis:
            state_manager.set_run_analysis(True)


def _render_analysis_results(analysis, recommendation, state_manager):
    """Render the analysis results."""
    # Convert the analysis to a nicely formatted version for display
    formatted_analysis = format_analysis_text(analysis)
    
    # Display headline recommendation
    if recommendation and 'action' in recommendation:
        _render_recommendation_summary(recommendation)
        
    # Display the full analysis
    st.markdown(formatted_analysis)


def _render_recommendation_summary(recommendation):
    """Render the recommendation summary card."""
    action = recommendation['action']
    confidence = recommendation.get('confidence', 0) * 100
    strategy_name = recommendation.get('strategy', {}).get('name', 'N/A')
    
    if action == 'BUY':
        action_color = "🟢"
    elif action == 'SELL':
        action_color = "🔴"
    else:
        action_color = "🟡"
    
    html_content = f'<div style="background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 10px 0;">'
    html_content += f'<h4 style="margin: 0; color: #1f77b4;">📋 Analysis Summary</h4>'
    html_content += f'<p style="margin: 5px 0; font-size: 18px;">'
    html_content += f'<strong>Recommendation:</strong> {action_color} <strong>{action}</strong> | <strong>Strategy:</strong> {strategy_name}'
    html_content += f' | <strong>Confidence:</strong> {confidence:.0f}%</p>'
    html_content += '</div>'
    
    # Display the HTML content
    st.markdown(html_content, unsafe_allow_html=True)