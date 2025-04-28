from fpdf import FPDF
from PIL import Image

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
        # Get original image dimensions
        img = Image.open(chart_path)
        img_width, img_height = img.size

        # Set image display width (almost full page width)
        display_width = 190  # Width in mm

        # Calculate height proportionally to maintain aspect ratio
        aspect_ratio = img_height / img_width
        display_height = display_width * aspect_ratio

        # Had to add this bc there wasn't enough space for all the charts 
        # Check if there's enough space for the image
        bottom_margin = 15 # same as footer margin
        available_space = self.h - self.get_y() - bottom_margin

        if display_height > available_space:
            self.add_page

        # Insert the image
        self.image(chart_path, x=10, y=self.get_y(), w=display_width, h=display_height)

        # Move the Y position after the image
        self.set_y(self.get_y() + display_height + 5)  # Add small padding (5mm)

    def add_analysis_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
