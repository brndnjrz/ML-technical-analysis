from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 18)
        self.cell(0, 10, "AI-Powered Stock Analysis", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_chart(self, chart_path):
        self.image(chart_path, x=10, y=self.get_y(), w=190)
        self.ln(125)  # Adjust depending on chart height

    def add_analysis_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
