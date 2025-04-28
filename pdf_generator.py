from fpdf import FPDF
from PIL import Image

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 18)
        self.cell(0, 10, "AI-Powered Stock Analysis", ln=True, align="C")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_chart(self, chart_path):
        # Get original image dimensions
        img = Image.open(chart_path)
        img_width, img_height = img.size

        # Set image display width (almost full page width)
        max_display_width = 190  # Max width in mm
        bottom_margin = 15 # same as footer
        available_space = self.h - self.get_y() - bottom_margin # max height in mm 

        # Calculate scaling factors
        scale_w = max_display_width / img_width
        scale_h = available_space / img_height

        # Use the smaller scale to fit both width and height
        scale = min(scale_w, scale_h)

        # Calculate final image size
        final_width = img_width * scale 
        final_height = img_height * scale

        # Center the image horizontally 
        x_position = (self.w - final_width) / 2

        # Insert the scaled image 
        self.image(chart_path, x=x_position, y=self.get_y(), w=final_width, h=final_height)

        # Move Y position after the image
        self.set_y(self.get_y() + final_height * 5)


    def add_analysis_text(self, text):
        text = text.replace("*", "").replace("#", "")
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
