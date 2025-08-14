import ollama
import base64
import io
import tempfile
import threading
import queue
from ..ai_agents import HedgeFundAI
import pandas as pd
import json
import time
import logging

# Setup logger for AI analysis
logger = logging.getLogger(__name__)

def format_recommendation_summary(recommendation: dict, options_priority: bool = False) -> str:
    """Format the AI recommendation into a readable summary"""
    try:
        market = recommendation.get('market_analysis', {})
        strategy = recommendation.get('strategy', {})
        signals = recommendation.get('signals', {})
        risk = recommendation.get('risk_assessment', {})
        consensus = recommendation.get('consensus_details', {})
        
        # Add options priority info
        options_priority_info = recommendation.get('options_priority', options_priority)
        
        # Safe formatting with None checks
        def safe_format(value, format_type='str', default='N/A'):
            if value is None:
                return default
            try:
                if format_type == 'float':
                    return f"{float(value):.1f}"
                elif format_type == 'percent':
                    return f"{float(value)*100:.0f}%"
                elif format_type == 'price':
                    return f"${float(value):.2f}"
                else:
                    return str(value).title()
            except (ValueError, TypeError):
                return default
        
        # Build consensus information section
        consensus_section = ""
        if consensus:
            agreement_score = consensus.get('agreement_score', 0)
            consensus_reached = consensus.get('consensus_reached', False)
            status_icon = "✅" if consensus_reached else "⚠️"
            
            consensus_section = f"""
🏦 Hedge Fund Consensus:
   • Agreement Score: {agreement_score:.1%}
   • Status: {status_icon} {'CONSENSUS REACHED' if consensus_reached else 'CONFLICTS RESOLVED'}
   • Committee Decision: {consensus.get('final_decision', 'HOLD').upper()}"""
            
            if not consensus_reached and consensus.get('conflicts'):
                consensus_section += f"\\n   • Conflicts Addressed: {len(consensus.get('conflicts', []))} strategy disagreements"
        
        summary = f"""
🤖 AI HEDGE FUND ANALYSIS SUMMARY
{'='*50}{consensus_section}

📊 Market Analysis:
   • RSI: {safe_format(market.get('RSI'), 'float', '0.0')} ({safe_format(market.get('momentum', {}).get('rsi', {}).get('condition'))})
   • MACD: {safe_format(market.get('MACD_Signal'))} trend
   • Volume: {safe_format(market.get('volume_signal'))}
   • Trend Strength (ADX): {safe_format(market.get('trend_strength'), 'float', '0.0')}

💡 Strategy Recommendation:
   • Strategy: {safe_format(strategy.get('name'))}
   • Action: {safe_format(recommendation.get('action'))}
   • Confidence: {safe_format(strategy.get('confidence'), 'percent', '0%')}
   • Position Type: {safe_format(strategy.get('type')).upper()}
   {f"• Options Focus: {'✅ ENABLED' if options_priority_info else '❌ DISABLED'}" if 'options_priority_info' in locals() else ""}

🎯 Trade Parameters:
   • Entry Price: {safe_format(recommendation.get('entry_price'), 'price', '$0.00')}
   • Stop Loss: {safe_format(recommendation.get('stop_loss'), 'price', '$0.00')}
   • Take Profit: {safe_format(recommendation.get('take_profit'), 'price', '$0.00')}
   • Position Size: {safe_format(signals.get('position_size'), 'float', '0.0')} shares

⚠️ Risk Assessment:
   • Risk Level: {safe_format(risk.get('risk_level'))}
   • Max Loss: {safe_format(risk.get('factors', {}).get('max_loss'), 'percent', '0.0%')}
   • Portfolio Risk: {safe_format(risk.get('factors', {}).get('portfolio_risk'), 'percent', '0.0%')}
"""
        return summary
    except Exception as e:
        logger.error(f"Error formatting recommendation: {e}")
        return f"""
🤖 AI HEDGE FUND ANALYSIS SUMMARY
{'='*50}
❌ Error generating recommendation summary
   • Raw data available but formatting failed
   • Please check the detailed analysis below
"""

def run_ai_analysis(daily_fig, timeframe_fig, data: pd.DataFrame, ticker: str, prompt: str, vision_timeout: int = 120, options_priority: bool = True):
    """Run enhanced AI analysis using both vision model and agent system
    
    Args:
        daily_fig: Plotly figure for daily chart analysis
        timeframe_fig: Plotly figure for selected timeframe chart analysis
        data: Stock data DataFrame
        ticker: Stock symbol
        prompt: Analysis prompt
        vision_timeout: Timeout for vision analysis in seconds (default: 120)
        options_priority: Whether to prioritize options strategies (default: True)
    """
    
    print("🤖 AI HEDGE FUND ANALYSIS STARTING")
    print("=" * 50)
    
    # 1. Get AI Hedge Fund Consensus Analysis
    print("🏦 Running Hedge Fund AI Analysis...")
    print("   • Analyst Agent: Market condition assessment")
    print("   • Strategy Agent: Trade strategy evaluation") 
    print("   • Execution Agent: Risk and timing analysis")
    print("   • Building investment committee consensus...")
    
    # Check if we should prioritize options strategies (calls, puts, and iron condors)
    # Use the options_priority parameter from the UI checkbox
    if options_priority:
        print("📈 Prioritizing options strategies (calls, puts, and iron condors)")
        config = {
            'prioritize_options_strategies': True,
            'preferred_strategies': ['Day Trading Calls/Puts', 'Iron Condor', 'Butterfly Spread']
        }
    else:
        print("📈 Using balanced strategy mix (stocks and options)")
        config = {}
    
    ai_system = HedgeFundAI(config)
    current_price = data['Close'].iloc[-1]
    
    # Initialize options data
    options_data = {}
    
    # Try to get options data from session state if available
    try:
        import streamlit as st
        if 'options' in st.session_state and ticker in st.session_state['options']:
            options_data = st.session_state['options'][ticker]
            print("📊 Adding options metrics to analysis")
    except (ImportError, KeyError, TypeError):
        print("⚠️ No options data available for analysis")
    
    recommendation = ai_system.analyze_and_recommend(
        data, 
        ticker, 
        current_price, 
        options_priority, 
        options_data=options_data
    )
    
    # Show consensus details if available
    if 'consensus_details' in recommendation:
        consensus = recommendation['consensus_details']
        print(f"\n📋 CONSENSUS BUILDING RESULTS:")
        print(f"   • Agreement Score: {consensus.get('agreement_score', 0):.1%}")
        print(f"   • Consensus Threshold: {consensus.get('threshold', 0.6):.0%}")
        print(f"   • Decision Status: {'✅ CONSENSUS REACHED' if consensus.get('consensus_reached', False) else '⚠️ CONFLICT DETECTED'}")
        
        if not consensus.get('consensus_reached', False):
            conflicts = consensus.get('conflicts', [])
            if conflicts:
                print(f"   • Conflicts Resolved: {len(conflicts)} strategy conflicts addressed")
    
    # Log formatted summary instead of raw JSON
    recommendation['options_priority'] = options_priority  # Add options priority to recommendation
    summary = format_recommendation_summary(recommendation, options_priority)
    print(summary)

    # 2. Get Vision Model Analysis
    print("\n👁️ Starting visual chart analysis...")
    
    # Check if Ollama is available
    try:
        print("🔌 Checking Ollama connection...")
        # Try a simple ping to Ollama first
        test_response = ollama.list()
        print("✅ Ollama service is running")
        
        # Debug: print the response type (but not the full response to avoid clutter)
        print(f"🔍 Ollama response type: {type(test_response).__name__}")
        
        # Check if vision model is available with safer access
        available_models = []
        try:
            # Handle different response types from Ollama
            if hasattr(test_response, 'models'):
                # New Ollama response format with ListResponse object
                for model in test_response.models:
                    if hasattr(model, 'model'):
                        available_models.append(model.model)
                    elif hasattr(model, 'name'):
                        available_models.append(model.name)
                    else:
                        available_models.append(str(model))
            elif isinstance(test_response, dict):
                # Legacy dictionary format
                models_list = test_response.get('models', [])
                for model in models_list:
                    if isinstance(model, dict):
                        # Try different possible keys for model name
                        model_name = model.get('name') or model.get('model') or model.get('id') or str(model)
                        if model_name:
                            available_models.append(model_name)
                    else:
                        # Handle case where model is a string directly
                        available_models.append(str(model))
            else:
                print(f"⚠️ Unexpected response format: {type(test_response).__name__}")
                # Try to extract models from string representation as fallback
                response_str = str(test_response)
                if 'llama3.2-vision' in response_str:
                    available_models.append('llama3.2-vision:latest')
                
        except Exception as model_parse_error:
            print(f"⚠️ Error parsing models list: {model_parse_error}")
            # Fallback: try to extract from string representation
            response_str = str(test_response)
            if 'llama3.2-vision' in response_str:
                available_models.append('llama3.2-vision:latest')
        
        print(f"📋 Available models: {available_models}")
        
        # Check for vision model with more flexible matching
        vision_model_found = any(
            'llama3.2-vision' in model.lower() or 'vision' in model.lower() 
            for model in available_models
        )
        
        if not vision_model_found:
            print(f"⚠️ llama3.2-vision model not found. Available models: {available_models}")
            print("📋 Skipping vision analysis - using AI agent analysis only")
            vision_response = {'message': {'content': 'Vision analysis skipped. The llama3.2-vision model is not installed. Please install it with: ollama pull llama3.2-vision'}}
        else:
            print("✅ Vision model available")
            # Proceed with vision analysis
            
            # Suppress verbose Kaleido logging temporarily
            import logging as base_logging
            kaleido_logger = base_logging.getLogger('kaleido')
            original_level = kaleido_logger.level
            kaleido_logger.setLevel(base_logging.WARNING)
            
            try:
                # Create a combined image with both charts vertically stacked
                from PIL import Image
                import numpy as np
                
                # Render both figures to images with optimized smaller sizes for faster processing
                daily_buf = io.BytesIO()
                timeframe_buf = io.BytesIO()
                
                # Use smaller sizes to speed up processing (reduced from 1000x500 to 800x400)
                daily_fig.write_image(daily_buf, format='png', width=800, height=400, scale=1.0)
                timeframe_fig.write_image(timeframe_buf, format='png', width=800, height=400, scale=1.0)
                
                daily_buf.seek(0)
                timeframe_buf.seek(0)
                
                # Open images with PIL
                daily_img = Image.open(daily_buf)
                timeframe_img = Image.open(timeframe_buf)
                
                # Create a smaller combined image
                total_width = max(daily_img.width, timeframe_img.width)
                total_height = daily_img.height + timeframe_img.height + 15  # Reduced padding
                
                combined_img = Image.new('RGB', (total_width, total_height), color='white')
                
                # Paste the images
                combined_img.paste(daily_img, (0, 0))
                combined_img.paste(timeframe_img, (0, daily_img.height + 15))
                
                # Save combined image to buffer with optimization
                buf = io.BytesIO()
                combined_img.save(buf, format='PNG', optimize=True, quality=80)
                buf.seek(0)
                
                # Check image size and aggressively optimize if needed
                image_size = buf.getbuffer().nbytes
                print(f"📊 Combined chart image size: {image_size / 1024:.1f} KB")
                
                if image_size > 300 * 1024:  # Reduced threshold from 500KB to 300KB
                    print("🔧 Optimizing large image for faster processing...")
                    # More aggressive resizing
                    combined_img = combined_img.resize((int(combined_img.width * 0.6), int(combined_img.height * 0.6)))
                    buf = io.BytesIO()
                    combined_img.save(buf, format='PNG', optimize=True, quality=70)
                    buf.seek(0)
                    optimized_size = buf.getbuffer().nbytes
                    print(f"✅ Optimized image size: {optimized_size / 1024:.1f} KB")
                
                image_data = base64.b64encode(buf.read()).decode('utf-8')
                print("✅ Combined chart image prepared for AI vision analysis")
            except Exception as e:
                print(f"❌ Error preparing chart: {e}")
                return "Error in chart preparation", recommendation
            finally:
                # Restore original logging level
                kaleido_logger.setLevel(original_level)
                
    except Exception as ollama_check_error:
        print(f"❌ Ollama connection failed: {ollama_check_error}")
        print("📋 Skipping vision analysis - Ollama service unavailable")
        vision_response = {'message': {'content': 'Vision analysis unavailable. Ollama service is not running. Please start Ollama and ensure llama3.2-vision model is installed.'}}
        
    # Only proceed with vision analysis if Ollama and model are available AND vision is enabled
    if 'image_data' in locals() and vision_timeout > 0:
        # Create optimized prompt with agent insights (much shorter for faster processing)
        enhanced_prompt = f"""Analyze these {ticker} stock charts quickly:

TOP: Daily chart (trend context)
BOTTOM: Detailed chart (recent signals)

Current data: RSI {recommendation['market_analysis'].get('RSI', 0):.1f}, {recommendation.get('action', 'HOLD')} recommended

Provide brief analysis:
1. Trend direction (bullish/bearish/neutral)
2. Key support/resistance levels
3. Entry signals visible
4. Risk assessment

Be concise and focus on actionable insights."""
        
        # Get vision model analysis
        messages = [{
            'role': 'user',
            'content': enhanced_prompt,
            'images': [image_data]
        }]
        
        print("🔍 Processing with AI vision model...")
        start_time = time.time()
        
        try:
            # Use a simple, direct approach to avoid threading complexity and signal issues
            print("🔄 Connecting to Ollama vision model...")
            
            try:
                # Attempt direct connection without complex threading
                vision_response = ollama.chat(
                    model='llama3.2-vision', 
                    messages=messages, 
                    stream=False
                )
                
                duration = time.time() - start_time
                print(f"✅ Vision analysis completed in {duration:.1f}s")
                
            except Exception as e:
                print(f"⚠️ Primary vision analysis failed: {e}")
                
                # Try simplified fallback immediately
                try:
                    print("🔄 Attempting simplified analysis...")
                    simple_messages = [{
                        'role': 'user',
                        'content': f'Brief {ticker} chart trend analysis.',
                        'images': [image_data]
                    }]
                    
                    vision_response = ollama.chat(
                        model='llama3.2-vision', 
                        messages=simple_messages,
                        stream=False
                    )
                    
                    duration = time.time() - start_time
                    print(f"✅ Simplified vision analysis completed in {duration:.1f}s")
                    
                except Exception as fallback_error:
                    print(f"❌ All vision analysis attempts failed: {fallback_error}")
                    vision_response = {
                        'message': {
                            'content': 'Vision analysis unavailable. Using AI agent analysis for trading insights.'
                        }
                    }
                
        except Exception as e:
            print(f"❌ Error in vision analysis: {e}")
            vision_response = {
                'message': {
                    'content': 'Vision analysis failed due to connection issues. Using AI agent analysis for trading insights.'
                }
            }
    else:
        # Vision analysis is disabled or unavailable
        if vision_timeout == 0:
            print("📋 Vision analysis disabled by user - using AI agent analysis only")
            vision_response = {'message': {'content': 'Vision analysis disabled. Analysis based on quantitative indicators and AI agent recommendations above.'}}
        else:
            print("📋 Vision analysis unavailable - using AI agent analysis only")
            vision_response = {'message': {'content': 'Vision analysis unavailable. Analysis based on quantitative indicators and AI agent recommendations above.'}}
    
    # Helper function to format trade parameters in a more readable way
    def format_trade_params(params):
        if not params:
            return "No parameters available"
            
        formatted_params = []
        for key, value in params.items():
            # Format the key nicely
            formatted_key = key.replace('_', ' ').title()
            
            # Format the value based on its type
            if isinstance(value, dict):
                # Handle nested dictionaries (like risk_management)
                formatted_params.append(f"• {formatted_key}:")
                for subkey, subvalue in value.items():
                    formatted_subkey = subkey.replace('_', ' ').title()
                    if isinstance(subvalue, list):
                        formatted_params.append(f"  • {formatted_subkey}:")
                        for item in subvalue:
                            formatted_params.append(f"    • {item}")
                    else:
                        formatted_params.append(f"  • {formatted_subkey}: {subvalue}")
            elif isinstance(value, list):
                # Handle lists
                formatted_params.append(f"• {formatted_key}:")
                for item in value:
                    formatted_params.append(f"  • {item}")
            elif isinstance(value, bool):
                # Format boolean values
                formatted_value = "✅ Yes" if value else "❌ No"
                formatted_params.append(f"• {formatted_key}: {formatted_value}")
            elif isinstance(value, (int, float)):
                # Format numeric values
                if 'price' in key.lower() or 'stop' in key.lower() or 'target' in key.lower():
                    formatted_params.append(f"• {formatted_key}: ${value:.2f}")
                else:
                    formatted_params.append(f"• {formatted_key}: {value}")
            else:
                # Format all other values
                formatted_params.append(f"• {formatted_key}: {value}")
                
        return '\n'.join(formatted_params)
    
    # Combine both analyses
    combined_analysis = f"""
    🤖 AI TRADING ANALYSIS
    
    📊 Market Analysis:
    - RSI: {recommendation['market_analysis'].get('RSI', 0):.2f}
    - MACD Signal: {recommendation['market_analysis'].get('MACD_Signal', 'N/A')}
    - Volume: {recommendation['market_analysis'].get('volume_signal', 'N/A')}
    - Trend Strength (ADX): {recommendation['market_analysis'].get('trend_strength', 0):.2f}
    
    💡 Strategy Recommendation:
    - Strategy: {recommendation['strategy'].get('name', 'N/A')}
    - Confidence: {recommendation['strategy'].get('confidence', 0) * 100:.0f}%
    - Risk Level: {recommendation['risk_assessment'].get('risk_level', 'N/A')}
    
    📈 Trade Parameters:
    {format_trade_params(recommendation['strategy'].get('parameters', {}))}
    
    👁️ Visual Analysis:
    {vision_response['message']['content']}
    
    ⚠️ Risk Warning:
    This is AI-generated analysis for educational purposes only.
    Always conduct your own research and risk assessment.
    """
    
    print("\n" + "=" * 50)
    print("🎯 AI ANALYSIS COMPLETED")
    
    return combined_analysis, recommendation
