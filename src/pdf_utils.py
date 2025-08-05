import base64
import os
import tempfile
import pandas as pd
import streamlit as st
from src import pdf_generator

def generate_and_display_pdf(ticker, strategy_type, options_strategy, data, analysis, chart_path, levels, options_data, active_indicators):
    pdf = pdf_generator.EnhancedPDF()
    pdf.add_page()
    pdf.add_header(ticker, strategy_type, options_strategy)
    pdf.add_chart(chart_path)
    pdf.add_analysis_text(analysis)
    pdf.add_indicator_summary(data, active_indicators)
    pdf.add_risk_analysis(data, levels, options_data)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf.output(tmp_pdf.name)
        pdf_file_path = tmp_pdf.name
    with open(pdf_file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.markdown("### 📄 Comprehensive Analysis Report")
        st.markdown(pdf_display, unsafe_allow_html=True)
    with open(pdf_file_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Full Report",
            data=f.read(),
            file_name=f'{ticker}_{strategy_type.replace(" ", "_")}_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.pdf',
            mime="application/pdf"
        )
    os.remove(chart_path)
    os.remove(pdf_file_path)
