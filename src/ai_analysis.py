import ollama
import base64
import io
import tempfile
import threading
import queue
from src.ai_agents import HedgeFundAI
import pandas as pd
import json
import time
import logging

# Setup logger for AI analysis
logger = logging.getLogger(__name__)

def format_recommendation_summary(recommendation: dict) -> str:
    """Format the AI recommendation into a readable summary"""
    try:
        market = recommendation.get('market_analysis', {})
        strategy = recommendation.get('strategy', {})
        signals = recommendation.get('signals', {})
        risk = recommendation.get('risk_assessment', {})
        
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
        
        summary = f"""
🤖 AI ANALYSIS SUMMARY
{'='*50}
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
🤖 AI ANALYSIS SUMMARY
{'='*50}
❌ Error generating recommendation summary
   • Raw data available but formatting failed
   • Please check the detailed analysis below
"""

def run_ai_analysis(fig, data: pd.DataFrame, ticker: str, prompt: str):
    """Run enhanced AI analysis using both vision model and agent system"""
    
    print("🤖 AI ANALYSIS STARTING")
    print("=" * 50)
    
    # 1. Get AI Agents Analysis
    print("📊 Analyzing market conditions...")
    ai_system = HedgeFundAI()
    current_price = data['Close'].iloc[-1]
    
    recommendation = ai_system.analyze_and_recommend(data, ticker, current_price)
    
    # Log formatted summary instead of raw JSON
    summary = format_recommendation_summary(recommendation)
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
                buf = io.BytesIO()
                fig.write_image(buf, format='png')
                buf.seek(0)
                image_data = base64.b64encode(buf.read()).decode('utf-8')
                print("✅ Chart image prepared for AI vision analysis")
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
        
    # Only proceed with vision analysis if Ollama and model are available
    if 'image_data' in locals():
        # Create enhanced prompt with agent insights
        enhanced_prompt = f"""
        {prompt}
        
        AI Agent Analysis Summary:
        - Market Conditions: RSI {recommendation['market_analysis'].get('RSI', 0):.1f}, MACD {recommendation['market_analysis'].get('MACD_Signal', 'N/A')}, Volume {recommendation['market_analysis'].get('volume_signal', 'N/A')}
        - Recommended Strategy: {recommendation['strategy'].get('name', 'N/A')}
        - Confidence: {recommendation['strategy'].get('confidence', 0)*100:.0f}%
        - Action: {recommendation.get('action', 'HOLD')}
        
        Please analyze this chart and provide insights considering the above analysis.
        """
        
        # Get vision model analysis
        messages = [{
            'role': 'user',
            'content': enhanced_prompt,
            'images': [image_data]
        }]
        
        print("🔍 Processing with AI vision model...")
        start_time = time.time()
        
        try:
            # Use threading-based timeout instead of signal (Streamlit-compatible)
            import threading
            import queue
            
            def run_ollama_chat():
                """Run Ollama chat in a separate thread"""
                try:
                    result = ollama.chat(model='llama3.2-vision', messages=messages)
                    return result
                except Exception as e:
                    return {'error': str(e)}
            
            print("🔄 Connecting to Ollama vision model...")
            
            # Create a queue to get the result from the thread
            result_queue = queue.Queue()
            
            def worker():
                result = run_ollama_chat()
                result_queue.put(result)
            
            # Start the worker thread
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            
            # Wait for result with timeout
            try:
                vision_response = result_queue.get(timeout=60)  # 60 second timeout
                if isinstance(vision_response, dict) and 'error' in vision_response:
                    raise Exception(vision_response['error'])
                    
                duration = time.time() - start_time
                print(f"✅ Vision analysis completed in {duration:.1f}s")
                
            except queue.Empty:
                print("⏰ Vision analysis timed out after 60 seconds")
                vision_response = {'message': {'content': 'Vision analysis timed out. The chart shows technical indicators that can be analyzed manually. Please refer to the AI agent analysis above for trading insights.'}}
            except Exception as ollama_error:
                print(f"❌ Ollama connection error: {ollama_error}")
                vision_response = {'message': {'content': 'Vision analysis unavailable. Ollama service may not be running or the llama3.2-vision model may not be installed. Please refer to the AI agent analysis above for trading insights.'}}
                
        except Exception as e:
            print(f"❌ Error in vision analysis: {e}")
            # Simple fallback without timeout for maximum compatibility
            try:
                print("🔄 Attempting simple connection to Ollama...")
                vision_response = ollama.chat(model='llama3.2-vision', messages=messages)
                duration = time.time() - start_time
                print(f"✅ Vision analysis completed in {duration:.1f}s")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                vision_response = {'message': {'content': 'Vision analysis failed due to connection issues. Please refer to the AI agent analysis above for trading insights.'}}
    
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
    {json.dumps(recommendation['strategy'].get('parameters', {}), indent=2)}
    
    👁️ Visual Analysis:
    {vision_response['message']['content']}
    
    ⚠️ Risk Warning:
    This is AI-generated analysis for educational purposes only.
    Always conduct your own research and risk assessment.
    """
    
    print("\n" + "=" * 50)
    print("🎯 AI ANALYSIS COMPLETED")
    
    return combined_analysis, recommendation
