import base64
import os
import tempfile
import pandas as pd
import streamlit as st
import re
from src import pdf_generator

def sanitize_text(text):
    """Remove emojis and non-Latin characters from text for PDF compatibility."""
    # Remove emojis and other special characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        u"\U0001FA00-\U0001FA6F"  # extended symbols
        u"\U0001F916"             # robot face
        "]+", flags=re.UNICODE)
    
    # Replace emojis with their text equivalents
    replacements = {
        "🤖": "[AI]",
        "📊": "[Chart]",
        "💡": "[Insight]",
        "📈": "[Trend]",
        "🔍": "[Analysis]",
        "⚠️": "[Warning]",
        "✅": "[Success]",
        "❌": "[Fail]",
        "🟢": "[Long]",
        "🔴": "[Short]",
        "🔵": "[Neutral]"
    }
    
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    
    # Remove any remaining emojis
    text = emoji_pattern.sub(r'', text)
    
    # Replace other problematic characters
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace("'", "'")
    text = text.replace("…", "...")
    
    return text

def generate_and_display_pdf(ticker, strategy_type, options_strategy, data, analysis, chart_path, levels, options_data, active_indicators):
    temp_dir = None
    try:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, f"{ticker}_analysis.pdf")
        
        # Sanitize the analysis text
        if isinstance(analysis, dict):
            sanitized_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, str):
                    sanitized_analysis[key] = sanitize_text(value)
                elif isinstance(value, list):
                    sanitized_analysis[key] = [sanitize_text(item) if isinstance(item, str) else item for item in value]
                else:
                    sanitized_analysis[key] = value
        elif isinstance(analysis, str):
            sanitized_analysis = sanitize_text(analysis)
        else:
            sanitized_analysis = analysis
        
        # Generate PDF
        success = pdf_generator.generate_pdf(
            ticker=ticker,
            strategy_type=strategy_type,
            options_strategy=options_strategy,
            data=data,
            analysis=sanitized_analysis,
            chart_path=chart_path,
            levels=levels,
            options_data=options_data,
            active_indicators=active_indicators,
            output_path=pdf_path
        )
        
        if not success:
            st.error("Failed to generate PDF report.")
            return
            
        # Read PDF file and create download link
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        # Create a download button
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{ticker}_analysis.pdf",
            mime="application/pdf"
        )
        
        # Display PDF preview using base64 encoding
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        print(f"PDF Generation Error: {str(e)}")  # Console log for debugging
    finally:
        # Cleanup temporary files
        try:
            # Clean up the PDF file and its directory
            if temp_dir and os.path.exists(temp_dir):
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                os.rmdir(temp_dir)
            
            # Clean up the chart file
            if chart_path and os.path.exists(chart_path):
                os.remove(chart_path)
                # Try to remove the temp directory if it's empty
                chart_dir = os.path.dirname(chart_path)
                if os.path.exists(chart_dir) and not os.listdir(chart_dir):
                    os.rmdir(chart_dir)
        except Exception as e:
            print(f"Cleanup Error: {str(e)}")  # Console log for debugging
