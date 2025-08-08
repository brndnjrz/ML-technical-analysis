import ollama
import base64
import io
import tempfile
from src.ai_agents import HedgeFundAI
import pandas as pd
import json

def run_ai_analysis(fig, data: pd.DataFrame, ticker: str, prompt: str):
    """Run enhanced AI analysis using both vision model and agent system"""
    
    # 1. Get AI Agents Analysis
    ai_system = HedgeFundAI()
    current_price = data['Close'].iloc[-1]
    
    recommendation = ai_system.analyze_and_recommend(data, ticker, current_price)
    
    # 2. Get Vision Model Analysis
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create enhanced prompt with agent insights
    enhanced_prompt = f"""
    {prompt}
    
    AI Agent Analysis Summary:
    - Market Conditions: {json.dumps(recommendation['market_analysis'], indent=2)}
    - Recommended Strategy: {recommendation['strategy']['name']}
    - Confidence: {recommendation['strategy']['confidence']}
    - Reasoning: {', '.join(recommendation['strategy']['reason'])}
    
    Please analyze this chart and provide insights considering the above analysis.
    """
    
    # Get vision model analysis
    messages = [{
        'role': 'user',
        'content': enhanced_prompt,
        'images': [image_data]
    }]
    vision_response = ollama.chat(model='llama3.2-vision', messages=messages)
    
    # Combine both analyses
    combined_analysis = f"""
    🤖 AI TRADING ANALYSIS
    
    📊 Market Analysis:
    - RSI: {recommendation['market_analysis']['RSI']:.2f}
    - MACD Signal: {recommendation['market_analysis']['MACD_Signal']}
    - Volume: {recommendation['market_analysis']['volume_signal']}
    - Trend Strength (ADX): {recommendation['market_analysis']['trend_strength']:.2f}
    
    💡 Strategy Recommendation:
    - Strategy: {recommendation['strategy']['name']}
    - Confidence: {recommendation['strategy']['confidence'] * 100:.0f}%
    - Risk Level: {recommendation['strategy']['risk_level']}
    
    📈 Trade Parameters:
    {json.dumps(recommendation['parameters'], indent=2)}
    
    👁️ Visual Analysis:
    {vision_response['message']['content']}
    
    ⚠️ Risk Warning:
    This is AI-generated analysis for educational purposes only.
    Always conduct your own research and risk assessment.
    """
    
    return combined_analysis, recommendation
