"""AI analysis workflow management."""
import streamlit as st
import logging
import os
from ..utils.app_config import ProgressSteps, UIConfig
from ..utils.temp_manager import temp_manager
from ..utils.workflow_logger import (
    log_section_start, log_section_end, log_timer_start, log_timer_end,
    log_step
)
from .prediction import predict_next_period_close
from .ai_analysis import run_ai_analysis
from ..ai_agents.strategy_arbiter import choose_final_strategy
from ..utils.ai_output_schema import validate_ai_model_output


class AnalysisWorkflowManager:
    """Manages the AI analysis workflow process."""
    
    def __init__(self, state_manager):
        self.state_manager = state_manager
    
    def run_analysis_workflow(self, data, fundamentals, active_indicators, ticker, 
                            prompt, options_priority, candidate_strategies, features, 
                            user_timeframe, daily_fig, subplot_fig, interval='1d'):
        """Run the complete AI analysis workflow."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if 'prompt' not in locals() or prompt is None:
            prompt = "PROVIDE:"
        
        with st.spinner("🤖 AI is analyzing the market..."):
            print("\n" + "="*60)
            print("🤖 STARTING AI MARKET ANALYSIS")
            print("="*60)
            
            try:
                # Step 1: Price Prediction
                prediction_context = self._run_price_prediction(
                    data, fundamentals, active_indicators, interval, progress_bar, status_text
                )
                
                # Step 2: Prepare Chart
                chart_path = self._prepare_chart_analysis(
                    ticker, subplot_fig, progress_bar, status_text
                )
                
                # Step 3: Update prompt with prediction
                prompt = self._update_prompt_with_prediction(prompt, prediction_context)
                
                # Step 4: Run AI Analysis
                analysis, recommendation = self._run_ai_analysis(
                    daily_fig, data, ticker, prompt, options_priority, progress_bar, status_text
                )
                
                # Step 5: Strategy Arbitration
                final_strategy = self._run_strategy_arbitration(
                    recommendation, candidate_strategies, user_timeframe, features
                )
                
                # Step 6: Validation
                validated_strategy = self._validate_output(final_strategy, ticker)
                
                # Step 7: Complete
                self._complete_analysis(
                    analysis, chart_path, validated_strategy, ticker,
                    progress_bar, status_text
                )
                
                return True
                
            except Exception as e:
                self._handle_analysis_error(e, progress_bar, status_text)
                return False
    
    def _run_price_prediction(self, data, fundamentals, active_indicators, interval, progress_bar, status_text):
        """Run price prediction step."""
        status_text.text("🔮 Generating price predictions...")
        progress_bar.progress(ProgressSteps.PREDICTION)
        
        try:
            prediction_result = predict_next_period_close(
                data.copy(),
                fundamentals,
                active_indicators,
                interval
            )
            
            if prediction_result and isinstance(prediction_result, tuple) and prediction_result[0] is not None:
                predicted_price, confidence = prediction_result
                data['Predicted_Close'] = float(predicted_price)
                price_change = predicted_price - data['Close'].iloc[-1]
                data['Predicted_Price_Change'] = price_change
                
                prediction_context = f"""PREDICTED NEXT {interval.upper()} CLOSE: ${predicted_price:.2f} (Confidence: {confidence:.1%})\nPRICE CHANGE: ${price_change:.2f} ({(price_change/data['Close'].iloc[-1]*100):.1f}%)"""
                print(f"✅ Price prediction: ${predicted_price:.2f} (Confidence: {confidence:.1%})")
                return prediction_context
            else:
                print("⚠️ AI price prediction temporarily unavailable (insufficient data)")
                data_size = len(data)
                print(f"📊 Current dataset: {data_size} rows (minimum {UIConfig.MIN_DATASET_SIZE} recommended)")
                return f"AI PRICE PREDICTION: Unavailable due to small dataset ({data_size} rows). Consider using a longer date range."
                
        except Exception as e:
            print(f"⚠️ Prediction error: {str(e)}")
            return "AI PRICE PREDICTION: Temporarily unavailable"
    
    def _prepare_chart_analysis(self, ticker, subplot_fig, progress_bar, status_text):
        """Prepare chart for AI analysis."""
        status_text.text("📊 Preparing chart analysis...")
        progress_bar.progress(ProgressSteps.CHART_PREP)
        
        chart_path = temp_manager.create_chart_file(ticker)
        
        # Temporarily suppress kaleido logging
        kaleido_logger = logging.getLogger('kaleido')
        original_level = kaleido_logger.level
        kaleido_logger.setLevel(logging.WARNING)
        
        try:
            subplot_fig.write_image(chart_path)
            print(f"✅ Chart prepared for AI analysis")
            return chart_path
        except Exception as e:
            print(f"❌ Error saving chart: {e}")
            st.error(f"Failed to save chart: {e}")
            return None
        finally:
            kaleido_logger.setLevel(original_level)
    
    def _update_prompt_with_prediction(self, prompt, prediction_context):
        """Update prompt with prediction context."""
        if "Short-Term" in prompt:
            prompt = prompt.replace("PROVIDE:", f"{prediction_context}\nCONSIDER HOW THIS PRICE AFFECTS SHORT-TERM INDICATORS AND MOMENTUM.\n\nPROVIDE:")
        else:
            prompt = prompt.replace("PROVIDE:", f"{prediction_context}\nCONSIDER HOW THIS PRICE AFFECTS TRENDS AND LONG-TERM STRATEGIES.\n\nPROVIDE:")
        
        return prompt
    
    def _run_ai_analysis(self, daily_fig, data, ticker, prompt, options_priority, progress_bar, status_text):
        """Run the AI analysis step."""
        status_text.text("🧠 Running AI analysis...")
        progress_bar.progress(ProgressSteps.AI_ANALYSIS)
        
        # Start timer for overall AI analysis performance tracking
        ai_analysis_timer = log_timer_start("AI Analysis Execution")
        log_section_start(f"AI MARKET ANALYSIS FOR {ticker}")
        
        try:
            analysis, recommendation = run_ai_analysis(
                daily_fig=daily_fig,
                timeframe_fig=None,
                data=data,
                ticker=ticker,
                prompt=prompt,
                options_priority=options_priority
            )
            
            log_timer_end("AI Analysis Execution", ai_analysis_timer)
            return analysis, recommendation
            
        finally:
            log_section_end(f"AI MARKET ANALYSIS FOR {ticker}")
    
    def _run_strategy_arbitration(self, recommendation, candidate_strategies, user_timeframe, features):
        """Run strategy arbitration to select final strategy."""
        # Add LLM/AI output as a candidate
        if recommendation:
            candidate_strategies.append({
                "name": recommendation.get('strategy', {}).get('name', recommendation.get('action', 'Unknown')),
                "timeframe": user_timeframe,
                "trend": features.get('trend', None),
                "type": recommendation.get('strategy', {}).get('name', '').lower().replace(' ', '_'),
                "iv_rank": features.get('iv_rank', 0),
                "adx": features.get('adx', 0),
                "rsi": features.get('rsi', 50),
                "confidence": recommendation.get('strategy', {}).get('confidence', 0.7),
                "rationale": recommendation.get('strategy', {}).get('rationale', '')
            })
        
        # Use strategy arbiter to select final strategy
        return choose_final_strategy(candidate_strategies, user_timeframe, features)
    
    def _validate_output(self, final_strategy, ticker):
        """Validate the AI output."""
        try:
            log_step("Validating AI output schema", emoji="🔍")
            validation_result = validate_ai_model_output(final_strategy, ticker=ticker)
            
            if validation_result and not isinstance(validation_result, bool):
                # If validation returned a transformed object, use it
                final_strategy = validation_result
                log_step("AI output was automatically adapted to match the required schema", emoji="ℹ️")
                st.info("ℹ️ AI output was automatically adapted to match the required schema")
            else:
                log_step("AI output validation passed", emoji="✅")
                
        except Exception as schema_error:
            log_step(f"AI output failed schema validation: {schema_error}", emoji="⚠️")
            st.warning(f"⚠️ AI output failed schema validation: {schema_error}")
            logging.error(f"Schema validation error: {schema_error}\nData: {final_strategy}")
        
        return final_strategy
    
    def _complete_analysis(self, analysis, chart_path, final_strategy, ticker, progress_bar, status_text):
        """Complete the analysis workflow."""
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(ProgressSteps.COMPLETION)
        
        self.state_manager.set_ai_analysis_result(analysis, chart_path, final_strategy)
        self.state_manager.set_run_analysis(False)
        self.state_manager.set_analysis_running(False)
        
        # Clean up progress UI
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        print(f"✅ Analysis workflow completed for {ticker}")
    
    def _handle_analysis_error(self, error, progress_bar, status_text):
        """Handle analysis workflow errors."""
        status_text.text("❌ Analysis failed")
        progress_bar.progress(0)
        st.error(f"AI analysis failed: {error}")
        
        import traceback
        traceback.print_exc()
        
        self.state_manager.set_run_analysis(False)
        self.state_manager.set_analysis_running(False)
        
        # Clean up progress UI
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()