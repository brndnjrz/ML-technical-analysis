from fpdf import FPDF
from PIL import Image


# EnhancedPDF class with required methods for app.py
class EnhancedPDF(FPDF):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.add_font('Arial', '', 'arial.ttf', uni=True)  # Load Arial with unicode support
    #     self.add_font('Arial', 'B', 'arialbd.ttf', uni=True)  # Load Arial Bold with unicode support
    #     self.set_font('Arial', '', 12)  # Set default font

    def header(self):
        self.set_font("Arial", "B", 18)
        self.cell(0, 10, "AI-Powered Stock Analysis", ln=True, align="C")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_header(self, ticker, strategy_type, options_strategy=None):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, f"Ticker: {ticker}", ln=True)
        self.set_font("Arial", "", 12)
        self.cell(0, 8, f"Strategy Type: {strategy_type}", ln=True)
        if options_strategy:
            self.cell(0, 8, f"Options Strategy: {options_strategy}", ln=True)
        self.ln(4)

    def add_chart(self, chart_path):
        img = Image.open(chart_path)
        img_width, img_height = img.size
        max_display_width = 190
        bottom_margin = 15
        available_space = self.h - self.get_y() - bottom_margin
        scale_w = max_display_width / img_width
        scale_h = available_space / img_height
        scale = min(scale_w, scale_h)
        final_width = img_width * scale
        final_height = img_height * scale
        x_position = (self.w - final_width) / 2
        self.image(chart_path, x=x_position, y=self.get_y(), w=final_width, h=final_height)
        self.set_y(self.get_y() + final_height * 1.1)

    def add_analysis_text(self, text):
        text = text.replace("*", "").replace("#", "")
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
        self.ln(2)

    def add_indicator_summary(self, data, active_indicators):
        self.set_font("Arial", "B", 13)
        self.cell(0, 10, "Technical Indicator Summary", ln=True)
        self.set_font("Arial", "", 11)
        for ind in active_indicators:
            val = data[ind].iloc[-1] if ind in data.columns else "N/A"
            self.cell(0, 8, f"{ind}: {val}", ln=True)
        self.ln(2)

    def add_risk_analysis(self, data, levels, options_data=None):
        self.set_font("Arial", "B", 13)
        self.cell(0, 10, "Risk & Support/Resistance Analysis", ln=True)
        self.set_font("Arial", "", 11)
        # Support/Resistance
        support = levels.get('support', [])
        resistance = levels.get('resistance', [])
        self.cell(0, 8, f"Support Levels: {', '.join([f'${s:.2f}' for s in support]) if support else 'N/A'}", ln=True)
        self.cell(0, 8, f"Resistance Levels: {', '.join([f'${r:.2f}' for r in resistance]) if resistance else 'N/A'}", ln=True)
        # Options data
        if options_data and options_data.get('iv_data'):
            iv = options_data['iv_data']
            self.cell(0, 8, f"IV Rank: {iv.get('iv_rank', 'N/A')}%", ln=True)
            self.cell(0, 8, f"IV Percentile: {iv.get('iv_percentile', 'N/A')}%", ln=True)
            self.cell(0, 8, f"30-Day HV: {iv.get('hv_30', 'N/A')}%", ln=True)
            self.cell(0, 8, f"VIX Level: {iv.get('vix', 'N/A')}", ln=True)
        self.ln(2)

# For backward compatibility if needed
PDF = EnhancedPDF
