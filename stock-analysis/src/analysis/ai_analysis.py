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
from .vision_schema import parse_vision_analysis, create_vision_prompt_template
from ..utils.options_strategy_cheatsheet import OPTIONS_STRATEGY_CHEATSHEET
from ..utils.formatters import format_trade_params

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
   • RSI: {safe_format(market.get('RSI'), 'float', '0.0')} ({safe_format(market.get('momentum', dict()).get('rsi', dict()).get('condition'))})
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
   • Max Loss: {safe_format(risk.get('factors', dict()).get('max_loss'), 'percent', '0.0%')}
   • Portfolio Risk: {safe_format(risk.get('factors', dict()).get('portfolio_risk'), 'percent', '0.0%')}
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
    
    # Import workflow logger for enhanced logging
    from ..utils.workflow_logger import (log_section_start, log_section_end, 
                                         log_step, log_timer_start, log_timer_end)
    
    log_section_start(f"AI ANALYSIS FOR {ticker}")
    ai_start_time = log_timer_start("AI Analysis")
    
    # 1. Get AI Hedge Fund Consensus Analysis
    log_step("Running Hedge Fund AI Analysis", emoji="🏦")
    log_step("   • Analyst Agent: Market condition assessment")
    log_step("   • Strategy Agent: Trade strategy evaluation") 
    log_step("   • Execution Agent: Risk and timing analysis")
    log_step("   • Building investment committee consensus...")
    
    # Check if we should prioritize options strategies (calls, puts, and iron condors)
    # Use the options_priority parameter from the UI checkbox
    if options_priority:
        log_step("Prioritizing options strategies (calls, puts, and iron condors)", emoji="📈")
        config = {
            'prioritize_options_strategies': True,
            'preferred_strategies': ['Day Trading Calls/Puts', 'Iron Condors', 'Credit Spreads']
        }
    else:
        log_step("Using balanced strategy mix (stocks and options)", emoji="📈")
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
            log_step("Adding options metrics to analysis", emoji="📊")
    except (ImportError, KeyError, TypeError):
        log_step("No options data available for analysis", emoji="⚠️")
    
    hedge_fund_start = log_timer_start("Hedge Fund AI Analysis")
    recommendation = ai_system.analyze_and_recommend(
        data, 
        ticker, 
        current_price, 
        options_priority, 
        options_data=options_data
    )
    log_timer_end("Hedge Fund AI Analysis", hedge_fund_start)
    
    # Show consensus details if available
    if 'consensus_details' in recommendation:
        consensus = recommendation['consensus_details']
        log_step("\nCONSENSUS BUILDING RESULTS:", emoji="📋")
        log_step(f"   • Agreement Score: {consensus.get('agreement_score', 0):.1%}")
        log_step(f"   • Consensus Threshold: {consensus.get('threshold', 0.6):.0%}")
        log_step(f"   • Decision Status: {'✅ CONSENSUS REACHED' if consensus.get('consensus_reached', False) else '⚠️ CONFLICT DETECTED'}")
        
        if not consensus.get('consensus_reached', False):
            conflicts = consensus.get('conflicts', [])
            if conflicts:
                log_step(f"   • Conflicts Resolved: {len(conflicts)} strategy conflicts addressed")
    
    # Log formatted summary instead of raw JSON
    recommendation['options_priority'] = options_priority  # Add options priority to recommendation
    summary = format_recommendation_summary(recommendation, options_priority)
    log_step(summary)

    # 2. Get Vision Model Analysis
    log_step("\nStarting visual chart analysis...", emoji="👁️")
    
    # Check if Ollama is available
    image_data = None
    try:
        log_step("Checking Ollama connection...", emoji="🔌")
        # Try a simple ping to Ollama first
        ollama_start = log_timer_start("Ollama Connection Test")
        test_response = ollama.list()
        log_timer_end("Ollama Connection Test", ollama_start)
        log_step("Ollama service is running", emoji="✅")
        
        # Debug: print the response type (but not the full response to avoid clutter)
        log_step(f"Ollama response type: {type(test_response).__name__}", emoji="🔍")
        
        available_models = []
        try:
            models_raw = getattr(test_response, 'models', None) or test_response.get('models', [])
            available_models = [
                getattr(m, 'model', None) or getattr(m, 'name', None) or str(m)
                for m in models_raw
            ]
            if not available_models and 'llama3.2-vision' in str(test_response):
                available_models = ['llama3.2-vision:latest']
        except Exception:
            if 'llama3.2-vision' in str(test_response):
                available_models = ['llama3.2-vision:latest']
        
        log_step(f"Available models: {available_models}", emoji="📋")
        
        # Check for vision model with more flexible matching
        vision_model_found = any(
            'llama3.2-vision' in model.lower() or 'vision' in model.lower() 
            for model in available_models
        )
        
        if not vision_model_found:
            log_step(f"llama3.2-vision model not found. Available models: {available_models}", emoji="⚠️")
            log_step("Skipping vision analysis - using AI agent analysis only", emoji="📋")
            vision_response = {'message': {'content': 'Vision analysis skipped. The llama3.2-vision model is not installed. Please install it with: ollama pull llama3.2-vision'}}
        else:
            log_step("Vision model available", emoji="✅")
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
                
                log_step("Preparing chart images for vision analysis...", emoji="🖼️")
                image_prep_start = log_timer_start("Image Preparation")
                
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
                log_step(f"Combined chart image size: {image_size / 1024:.1f} KB", emoji="📊")
                
                if image_size > 300 * 1024:  # Reduced threshold from 500KB to 300KB
                    log_step("Optimizing large image for faster processing...", emoji="🔧")
                    # More aggressive resizing
                    combined_img = combined_img.resize((int(combined_img.width * 0.6), int(combined_img.height * 0.6)))
                    buf = io.BytesIO()
                    combined_img.save(buf, format='PNG', optimize=True, quality=70)
                    buf.seek(0)
                    optimized_size = buf.getbuffer().nbytes
                    log_step(f"Optimized image size: {optimized_size / 1024:.1f} KB", emoji="✅")
                
                image_data = base64.b64encode(buf.read()).decode('utf-8')
                log_timer_end("Image Preparation", image_prep_start)
                log_step("Combined chart image prepared for AI vision analysis", emoji="✅")
            except Exception as e:
                log_step(f"Error preparing chart: {e}", emoji="❌")
                return "Error in chart preparation", recommendation
            finally:
                # Restore original logging level
                kaleido_logger.setLevel(original_level)
                
    except Exception as ollama_check_error:
        log_step(f"Ollama connection failed: {ollama_check_error}", emoji="❌")
        log_step("Skipping vision analysis - Ollama service unavailable", emoji="📋")
        vision_response = {'message': {'content': 'Vision analysis unavailable. Ollama service is not running. Please start Ollama and ensure llama3.2-vision model is installed.'}}
        
    # Only proceed with vision analysis if Ollama and model are available AND vision is enabled
    # if 'image_data' in locals() and vision_timeout > 0:
    if image_data is not None:
        # Create structured vision prompt
        current_price = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1] if 'ATR' in data.columns and len(data) > 0 else current_price * 0.02
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns and len(data) > 0 else 50
        iv_rank = options_data.get('iv_data', dict()).get('iv_rank', 0) if options_data else 0
        
        log_step("Creating structured vision prompt", emoji="✍️")
        structured_prompt = create_vision_prompt_template(
            ticker=ticker,
            timeframe="Daily/Intraday Combined",
            current_price=current_price,
            atr=atr,
            rsi=rsi,
            iv_rank=iv_rank
        )
        
        # Get vision model analysis
        messages = [{
            'role': 'user',
            'content': structured_prompt,
            'images': [image_data]
        }]
        
        log_step("Processing with AI vision model...", emoji="🔍")
        vision_start_time = log_timer_start("Vision Analysis")
        
        try:
            # Use a simple, direct approach to avoid threading complexity and signal issues
            log_step("Connecting to Ollama vision model...", emoji="🔄")
            
            try:
                # Attempt direct connection without complex threading
                vision_response = ollama.chat(
                    model='llama3.2-vision', 
                    messages=messages, 
                    stream=False
                )
                
                log_timer_end("Vision Analysis", vision_start_time)
                
                # Parse structured vision output
                log_step("Parsing vision analysis output", emoji="🔎")
                raw_vision_content = vision_response['message']['content']
                structured_vision = parse_vision_analysis(raw_vision_content, current_price)
                
                # Add parsed vision analysis to recommendation for fusion
                recommendation['vision_analysis'] = structured_vision
                log_step(f"Vision Analysis: {structured_vision.get('trend', 'neutral')} trend, "
                      f"confidence {structured_vision.get('confidence', 0):.2f}", emoji="📊")
                
            except Exception as e:
                log_step(f"Primary vision analysis failed: {e}", emoji="⚠️")
                
                # Try simplified fallback immediately
                try:
                    log_step("Attempting simplified analysis...", emoji="🔄")
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
                    
                    log_timer_end("Vision Analysis (Simplified)", vision_start_time)
                    
                except Exception as fallback_error:
                    log_step(f"All vision analysis attempts failed: {fallback_error}", emoji="❌")
                    vision_response = {
                        'message': {
                            'content': 'Vision analysis unavailable. Using AI agent analysis for trading insights.'
                        }
                    }
                
        except Exception as e:
            log_step(f"Error in vision analysis: {e}", emoji="❌")
            vision_response = {
                'message': {
                    'content': 'Vision analysis failed due to connection issues. Using AI agent analysis for trading insights.'
                }
            }

    # Combine both analyses with enhanced vision output
    log_step("Combining AI analysis results", emoji="🔄")
    vision_content = "Vision analysis unavailable"
    if 'vision_analysis' in recommendation:
        vision_analysis = recommendation['vision_analysis']
        if vision_analysis.get('schema_validation') == 'passed':
            support_str = ', '.join([f'${level:.2f}' for level in vision_analysis.get('support', [])])
            resistance_str = ', '.join([f'${level:.2f}' for level in vision_analysis.get('resistance', [])])
            patterns_str = ', '.join(vision_analysis.get('pattern_recognition', ['None']))
            
            vision_content = (
                f"📊 Structured Vision Analysis:\n"
                f"• Trend: {vision_analysis.get('trend', 'neutral').upper()}\n"
                f"• Confidence: {vision_analysis.get('confidence', 0)*100:.0f}%\n"
                f"• Support Levels: {support_str}\n"
                f"• Resistance Levels: {resistance_str}\n"
                f"• Risk Assessment: {vision_analysis.get('risk', 'medium').upper()}\n"
                f"• Patterns Detected: {patterns_str}"
            )
        else:
            # Fallback to raw content if available
            vision_content = "Vision analysis parsing failed - using raw output"
    else:
        vision_content = "Vision analysis was not performed or failed"
    
    # Build the combined analysis
    log_step("Generating final analysis report", emoji="📝")
    market_analysis = recommendation.get('market_analysis', {})
    strategy = recommendation.get('strategy', {})
    risk_assessment = recommendation.get('risk_assessment', {})
    
    combined_analysis = (
        f"🤖 AI TRADING ANALYSIS\n\n"
        f"📊 Market Analysis:\n"
        f"- RSI: {market_analysis.get('RSI', 0):.2f}\n"
        f"- MACD Signal: {market_analysis.get('MACD_Signal', 'N/A')}\n"
        f"- Volume: {market_analysis.get('volume_signal', 'N/A')}\n"
        f"- Trend Strength (ADX): {market_analysis.get('trend_strength', 0):.2f}\n\n"
        f"💡 Strategy Recommendation:\n"
        f"- Strategy: {strategy.get('name', 'N/A')}\n"
        f"- Confidence: {strategy.get('confidence', 0) * 100:.0f}%\n"
        f"- Risk Level: {risk_assessment.get('risk_level', 'N/A')}\n\n"
        f"📈 Trade Parameters:\n"
        f"{format_trade_params(strategy.get('parameters', dict()))}\n\n"
        f"👁️ Visual Analysis:\n"
        f"{vision_content}\n\n"
        f"⚠️ Risk Warning:\n"
        f"This is AI-generated analysis for educational purposes only.\n"
        f"Always conduct your own research and risk assessment."
    )
    
    log_timer_end("AI Analysis", ai_start_time)
    log_section_end(f"AI ANALYSIS FOR {ticker}")
    
    return combined_analysis, recommendation
