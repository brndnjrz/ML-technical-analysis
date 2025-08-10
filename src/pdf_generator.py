from fpdf import FPDF
from PIL import Image
import pandas as pd
import os


# EnhancedPDF class with required methods for app.py
class EnhancedPDF(FPDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use built-in fonts instead of trying to load external font files
        self.set_font('Arial', '', 12)  # Set default font
        
    def _escape(self, s):
        """Override the escape method to handle Unicode characters properly."""
        if isinstance(s, str):
            # Replace problematic Unicode characters with ASCII equivalents
            s = s.replace('\u2022', '-')  # Unicode bullet
            s = s.replace('•', '-')       # Bullet character  
            s = s.replace('–', '-')       # En dash
            s = s.replace('—', '-')       # Em dash
            s = s.replace('"', '"')       # Smart quotes
            s = s.replace('"', '"')
            s = s.replace(''', "'")       # Smart apostrophes
            s = s.replace(''', "'")
            s = s.replace('…', '...')     # Ellipsis
            # Convert to bytes using latin-1, replacing unsupported characters
            try:
                return s.encode('latin-1', errors='replace').decode('latin-1')
            except:
                # Fallback: remove any remaining problematic characters
                import re
                s = re.sub(r'[^\x00-\xFF]', '?', s)
                return s
        return s
    
    def safe_cell(self, w, h=0, txt='', border=0, ln=0, align='', fill=False, link=''):
        """Safe cell method that handles encoding issues."""
        if isinstance(txt, str):
            txt = self._escape(txt)
        return super().cell(w, h, txt, border, ln, align, fill, link)
        
    def safe_multi_cell(self, w, h, txt, border=0, align='J', fill=False):
        """Safe multi_cell method that handles encoding issues."""
        if isinstance(txt, str):
            txt = self._escape(txt)
        return super().multi_cell(w, h, txt, border, align, fill)

    def header(self):
        self.set_font("Arial", "B", 18)
        self.safe_cell(0, 10, "AI-Powered Stock Analysis", ln=True, align="C")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.safe_cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_header(self, ticker, strategy_type, options_strategy=None):
        self.set_font("Arial", "B", 14)
        self.safe_cell(0, 10, f"Ticker: {ticker}", ln=True)
        self.set_font("Arial", "", 12)
        self.safe_cell(0, 8, f"Strategy Type: {strategy_type}", ln=True)
        if options_strategy:
            self.safe_cell(0, 8, f"Options Strategy: {options_strategy}", ln=True)
        self.ln(4)

    def add_chart(self, chart_path):
        try:
            
            if not os.path.exists(chart_path):
                print(f"Warning: Chart file not found at {chart_path}")
                return
                
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
        except Exception as e:
            print(f"Error adding chart to PDF: {str(e)}")
            # Continue without the chart rather than failing
            self.safe_cell(0, 10, "Chart could not be added to the report", ln=True)

    def add_analysis_text(self, text):
        """Add formatted analysis text with proper structure and clean formatting."""
        if not text:
            self.safe_cell(0, 10, "No analysis available", ln=True)
            return
            
        # Clean up the text
        text = self.clean_analysis_text(text)
        
        # Split into sections for better formatting
        sections = text.split('\n\n')
        
        for section in sections:
            if section.strip():
                self.format_analysis_section(section.strip())
                
    def clean_analysis_text(self, text):
        """Clean and format analysis text for PDF compatibility."""
        # Fix character encoding issues
        text = text.replace('�', '[Analysis]')
        
        # Remove markdown formatting that doesn't work in PDF
        text = text.replace("*", "").replace("#", "")
        
        # Replace Unicode characters with ASCII equivalents
        text = text.replace('\u2022', '-')  # Unicode bullet to dash
        text = text.replace('•', '-')       # Bullet character to dash
        text = text.replace('–', '-')       # En dash to regular dash
        text = text.replace('—', '-')       # Em dash to regular dash
        text = text.replace('"', '"')       # Smart quotes to regular quotes
        text = text.replace('"', '"')
        text = text.replace(''', "'")       # Smart apostrophes to regular
        text = text.replace(''', "'")
        
        # Replace emojis with text equivalents
        emoji_replacements = {
            "🤖": "[AI Analysis]",
            "📊": "[Market Data]",
            "💡": "[Strategy]",
            "📈": "[Trade Parameters]",
            "👁️": "[Visual Analysis]",
            "⚠️": "[Risk Warning]",
            "🔍": "[Detailed Analysis]"
        }
        
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        
        return text
        
    def format_analysis_section(self, section):
        """Format individual analysis sections with proper headers and content."""
        lines = section.split('\n')
        
        if not lines:
            return
            
        # Check if this is a header section
        first_line = lines[0].strip()
        
        # Format headers
        if any(keyword in first_line for keyword in ['[AI Analysis]', '[Market Data]', '[Strategy]', '[Trade Parameters]', '[Visual Analysis]', '[Risk Warning]', '[Detailed Analysis]']):
            self.set_font("Arial", "B", 14)
            self.safe_cell(0, 12, first_line, ln=True)
            self.ln(2)
            
            # Format the content
            if len(lines) > 1:
                content = '\n'.join(lines[1:]).strip()
                self.format_section_content(content)
        else:
            # Regular content
            self.set_font("Arial", "", 11)
            self.safe_multi_cell(0, 8, section)
            self.ln(3)
            
    def format_section_content(self, content):
        """Format section content with proper bullet points and structure."""
        self.set_font("Arial", "", 11)
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle bullet points - use dash instead of Unicode bullet
            if line.startswith('- '):
                bullet_text = line[2:].strip()
                self.cell(5)  # indent
                self.safe_cell(0, 8, f"- {bullet_text}", ln=True)
            # Handle JSON-like parameters
            elif '{' in line and '}' in line:
                self.format_trade_parameters(line)
            else:
                self.safe_multi_cell(0, 8, line)
                
        self.ln(3)
        
    def format_trade_parameters(self, json_text):
        """Format trade parameters in a readable table format."""
        import json
        import re
        
        try:
            # Clean up the JSON text
            json_text = re.sub(r'^[^{]*{', '{', json_text)  # Remove text before {
            json_text = re.sub(r'}[^}]*$', '}', json_text)  # Remove text after }
            
            # Parse JSON
            params = json.loads(json_text)
            
            # Don't add another header since we already have one from the section
            self.set_font("Arial", "", 11)
            
            # Format parameters in a readable way
            for key, value in params.items():
                formatted_key = key.replace('_', ' ').title()
                
                if isinstance(value, bool):
                    formatted_value = "Yes" if value else "No"
                elif isinstance(value, (int, float)):
                    if 'price' in key.lower() or 'stop' in key.lower() or 'target' in key.lower():
                        formatted_value = f"${value:.2f}"
                    elif 'period' in key.lower():
                        formatted_value = f"{value} periods"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value).replace('_', ' ').title()
                
                self.cell(5)  # indent
                self.safe_cell(0, 8, f"- {formatted_key}: {formatted_value}", ln=True)
                
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to simple text if JSON parsing fails
            self.cell(5)  # indent
            self.safe_multi_cell(0, 8, f"Trade Parameters: {json_text}")
            
        self.ln(3)

    def add_indicator_summary(self, data, active_indicators):
        """Add technical indicators summary with improved formatting."""
        if data.empty:
            self.cell(0, 8, "No data available for indicators", ln=True)
            return
            
        self.set_font("Arial", "B", 13)
        self.cell(0, 10, "Current Market Indicators", ln=True)
        self.set_font("Arial", "", 11)
        self.ln(2)

        # Map indicator names to their corresponding DataFrame column names
        indicator_mapping = {
            # Regular Timeframe Indicators
            "RSI (14)": "RSI_14",
            "RSI (21)": "RSI_21",
            "RSI (Standard)": "RSI",
            "RSI (15min)": "RSI_5m",
            
            # Moving Averages
            "SMA (20)": "SMA_20",
            "SMA (50)": "SMA_50",
            "EMA (20)": "EMA_20",
            "EMA (50)": "EMA_50",
            "EMA (20) 15m": "EMA_20_5m",
            "EMA (50) 15m": "EMA_50_5m",
            "SMA (20) 15m": "SMA_20_5m",
            "SMA (50) 15m": "SMA_50_5m",
            
            # Volatility
            "ATR": "ATR",
            "ATR 15m": "ATR_5m",
            "Bollinger Bands": ["BB_upper", "BB_middle", "BB_lower"],
            "Bollinger Bands 15m": ["BB_upper_5m", "BB_middle_5m", "BB_lower_5m"],
            
            # Momentum
            "MACD": ["MACD", "MACD_Signal", "MACD_Hist"],
            "MACD 15m": ["MACD_5m", "MACD_Signal_5m", "MACD_Hist_5m"],
            "MACD Line": "MACD_Line",
            "Signal Line": "Signal_Line",
            "MACD Histogram": "MACD_Histogram",
            "Stochastic Fast 15m": ["STOCH_%K_5m", "STOCH_%D_5m"],
            
            # Volume
            "OBV": "OBV",
            "OBV 15m": "OBV_5m",
            "Volume": "Volume",
            "VWAP": "VWAP",
            "VWAP 15m": "VWAP_5m",
            
            # Trend
            "ADX": "ADX",
            "ADX 15m": "ADX_5m",
            
            # Predictions and Analysis
            "Predicted Close": "Predicted_Close",
            "Predicted Price Change": "Predicted_Price_Change",
            "Volatility": "volatility",
            "Returns": "returns"
        }

        # Group indicators by category for better organization
        indicator_categories = {
            "Momentum Indicators": ["RSI (14)", "RSI (21)", "MACD", "MACD Line", "Signal Line", "MACD Histogram", "Stochastic Fast 15m"],
            "Trend Indicators": ["SMA (20)", "SMA (50)", "EMA (20)", "EMA (50)", "ADX", "VWAP"],
            "Volatility Indicators": ["ATR", "Bollinger Bands", "Volatility"],
            "Volume Indicators": ["OBV", "Volume"],
            "Predictions": ["Predicted Close", "Predicted Price Change", "Returns"]
        }
        
        for category, indicators in indicator_categories.items():
            # Check if any indicators in this category are active
            category_indicators = [ind for ind in indicators if ind in active_indicators]
            
            if not category_indicators:
                continue
                
            # Category header
            self.set_font("Arial", "B", 12)
            self.cell(0, 8, category, ln=True)
            self.set_font("Arial", "", 10)
            
            for ind in category_indicators:
                try:
                    # Get the corresponding column name(s)
                    col_names = indicator_mapping.get(ind, ind)
                    
                    if isinstance(col_names, list):
                        # Handle multi-value indicators
                        values = []
                        for col in col_names:
                            if col in data.columns:
                                val = data[col].iloc[-1]
                                if not pd.isna(val):
                                    col_display = col.split('_')[-1] if '_' in col else col
                                    values.append(f"{col_display}: {val:.4f}")
                        val_str = " | ".join(values) if values else "N/A"
                        self.cell(5)  # indent
                        self.cell(0, 6, f"- {ind}: {val_str}", ln=True)
                    else:
                        # Handle single-value indicators
                        if col_names in data.columns:
                            val = data[col_names].iloc[-1]
                            if not pd.isna(val):
                                # Format based on indicator type
                                if any(x in ind for x in ["RSI", "ADX"]):
                                    val_str = f"{val:.2f}"
                                elif "Stochastic" in ind:
                                    val_str = f"{val:.2f}%"
                                elif any(x in ind for x in ["ATR", "SMA", "EMA", "VWAP", "Predicted Close"]):
                                    val_str = f"${val:.2f}"
                                elif "OBV" in ind:
                                    val_str = f"{val:,.0f}"
                                elif "Volatility" in ind:
                                    val_str = f"{val:.2%}"
                                elif "Returns" in ind:
                                    val_str = f"{val:.2%}"
                                elif "Predicted Price Change" in ind:
                                    val_str = f"${val:.2f}"
                                elif "Volume" in ind:
                                    val_str = f"{val:,.0f}"
                                else:
                                    val_str = f"{val:.4f}"
                            else:
                                val_str = "N/A"
                        else:
                            val_str = "N/A"
                        
                        self.cell(5)  # indent
                        self.cell(0, 6, f"- {ind}: {val_str}", ln=True)
                        
                except Exception as e:
                    print(f"Error processing indicator {ind}: {str(e)}")
                    self.cell(5)  # indent
                    self.cell(0, 6, f"- {ind}: Error", ln=True)
            
            self.ln(2)  # Space between categories
        
        self.ln(2)

    def add_risk_analysis(self, data, levels, options_data=None):
        """Add risk analysis section with enhanced formatting."""
        self.set_font("Arial", "B", 13)
        self.cell(0, 10, "Risk Assessment & Key Levels", ln=True)
        self.set_font("Arial", "", 11)
        self.ln(2)
        
        # Support/Resistance Levels
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, "Support & Resistance Levels", ln=True)
        self.set_font("Arial", "", 10)
        
        support = levels.get('support', [])
        resistance = levels.get('resistance', [])
        
        if support:
            support_str = ', '.join([f'${s:.2f}' for s in support])
            self.cell(5)  # indent
            self.cell(0, 6, f"- Support Levels: {support_str}", ln=True)
        else:
            self.cell(5)  # indent
            self.cell(0, 6, "- Support Levels: No clear support levels identified", ln=True)
            
        if resistance:
            resistance_str = ', '.join([f'${r:.2f}' for r in resistance])
            self.cell(5)  # indent
            self.cell(0, 6, f"- Resistance Levels: {resistance_str}", ln=True)
        else:
            self.cell(5)  # indent
            self.cell(0, 6, "- Resistance Levels: No clear resistance levels identified", ln=True)
        
        self.ln(3)
        
        # Options/Volatility Data
        if options_data and options_data.get('iv_data'):
            iv = options_data['iv_data']
            
            self.set_font("Arial", "B", 12)
            self.cell(0, 8, "Volatility & Options Metrics", ln=True)
            self.set_font("Arial", "", 10)
            
            metrics = [
                ("IV Rank", f"{iv.get('iv_rank', 'N/A')}%"),
                ("IV Percentile", f"{iv.get('iv_percentile', 'N/A')}%"),
                ("30-Day Historical Volatility", f"{iv.get('hv_30', 'N/A')}%"),
                ("VIX Level", f"{iv.get('vix', 'N/A')}")
            ]
            
            for metric_name, metric_value in metrics:
                self.cell(5)  # indent
                self.cell(0, 6, f"- {metric_name}: {metric_value}", ln=True)
        else:
            self.set_font("Arial", "B", 12)
            self.cell(0, 8, "Volatility Analysis", ln=True)
            self.set_font("Arial", "", 10)
            self.cell(5)  # indent
            self.cell(0, 6, "- Options data not available for detailed volatility analysis", ln=True)
        
        self.ln(3)
        
        # Risk Assessment based on available data
        if not data.empty:
            self.set_font("Arial", "B", 12)
            self.cell(0, 8, "Risk Assessment", ln=True)
            self.set_font("Arial", "", 10)
            
            try:
                current_price = data['Close'].iloc[-1]
                
                # Calculate basic risk metrics
                if 'volatility' in data.columns:
                    volatility = data['volatility'].iloc[-1]
                    if pd.notna(volatility):
                        risk_level = "High" if volatility > 0.4 else "Medium" if volatility > 0.2 else "Low"
                        self.cell(5)  # indent
                        self.cell(0, 6, f"- Volatility Risk: {risk_level} ({volatility:.1%} annualized)", ln=True)
                
                # Price level risk
                if support and resistance:
                    nearest_support = max([s for s in support if s < current_price], default=0)
                    nearest_resistance = min([r for r in resistance if r > current_price], default=float('inf'))
                    
                    if nearest_support > 0:
                        support_distance = ((current_price - nearest_support) / current_price) * 100
                        self.cell(5)  # indent
                        self.cell(0, 6, f"- Distance to Support: {support_distance:.1f}% (${nearest_support:.2f})", ln=True)
                    
                    if nearest_resistance != float('inf'):
                        resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
                        self.cell(5)  # indent
                        self.cell(0, 6, f"- Distance to Resistance: {resistance_distance:.1f}% (${nearest_resistance:.2f})", ln=True)
                
                # General risk note
                self.cell(5)  # indent
                self.cell(0, 6, "- Always use appropriate position sizing and stop-loss orders", ln=True)
                
            except Exception as e:
                self.cell(5)  # indent
                self.cell(0, 6, "- Risk metrics calculation error - use standard risk management", ln=True)
        
        self.ln(2)

    def add_indicators(self, data, active_indicators):
        """Add technical indicators section to the PDF."""
        if not active_indicators:
            return

        # Define indicator mappings
        indicator_mappings = {
            "RSI (14)": "RSI_14",
            "RSI (21)": "RSI_21",
            "SMA (20)": "SMA_20",
            "SMA (50)": "SMA_50",
            "EMA (20)": "EMA_20",
            "EMA (50)": "EMA_50",
            "ADX": "ADX",
            "ATR": "ATR",
            "Bollinger Bands Upper": "BB_upper",
            "Bollinger Bands Middle": "BB_middle",
            "Bollinger Bands Lower": "BB_lower",
            "OBV": "OBV",
            "VWAP": "VWAP",
            "Volume": "Volume",
            "MACD": "MACD",
            "MACD Line": "MACD_Line",
            "Signal Line": "MACD_Signal",
            "MACD Histogram": "MACD_Histogram",
            "Predicted Close": "predicted_close",
            "Predicted Price Change": "price_change"
        }

        # First check which indicators are active and in our data
        available_indicators = [name for name, col in indicator_mappings.items() 
                              if col in data.columns]
        
        # Debug print
        print("Available columns in DataFrame:", sorted(data.columns.tolist()))
        print("Mapped indicators found:", sorted(available_indicators))

        for display_name in available_indicators:
            try:
                column_name = indicator_mappings[display_name]
                val = data[column_name].iloc[-1]  # Get the most recent value
                
                if pd.notna(val):  # Check if the value is not NaN
                    # Format the value based on indicator type
                    if isinstance(val, (int, float)):
                        if "RSI" in column_name or "ADX" in column_name:
                            val_str = f"{val:.2f}"  # Regular format for RSI/ADX
                        elif "MACD" in column_name:
                            val_str = f"{val:.4f}"  # More precision for MACD
                        elif "MA" in column_name or "EMA" in column_name or "SMA" in column_name or "BB" in column_name:
                            val_str = f"${val:.2f}"  # Price format for moving averages and Bollinger Bands
                        elif column_name == "Volume":
                            val_str = f"{int(val):,}"  # Integer format for volume
                        elif "predicted" in column_name.lower():
                            val_str = f"${val:.2f}"  # Price format for predictions
                        elif "price_change" in column_name.lower():
                            val_str = f"${val:.2f}"  # Price format for changes
                        elif "%" in column_name or "Percent" in column_name:
                            val_str = f"{val:.2%}"  # Percentage format
                        else:
                            val_str = f"{val:.4f}"  # Default format
                    else:
                        val_str = str(val)
                else:
                    val_str = "N/A"
                
                self.cell(0, 8, f"{display_name}: {val_str}", ln=True)
            except Exception as e:
                print(f"Error processing indicator {display_name}: {str(e)}")
                self.cell(0, 8, f"{display_name}: Error", ln=True)
        
        self.ln(2)

# For backward compatibility if needed
PDF = EnhancedPDF

def generate_pdf(ticker, strategy_type, options_strategy, data, analysis, chart_path, levels, options_data, active_indicators, output_path):
    """Generate a PDF report with the analysis results."""
    pdf = EnhancedPDF()
    pdf.add_page()
    
    # Add header information
    pdf.add_header(ticker, strategy_type, options_strategy)
    
    # Add chart if available
    if chart_path:
        pdf.add_chart(chart_path)
    
    # Add analysis section with enhanced formatting
    pdf.set_font("Arial", "B", 16)
    pdf.safe_cell(0, 12, "AI Trading Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    # Handle different analysis formats
    if isinstance(analysis, dict):
        # Format dictionary analysis
        for key, value in analysis.items():
            pdf.set_font("Arial", "B", 13)
            formatted_key = key.replace('_', ' ').title()
            pdf.safe_cell(0, 10, formatted_key, ln=True)
            pdf.set_font("Arial", "", 11)
            
            if isinstance(value, list):
                for item in value:
                    pdf.cell(10)  # indent
                    pdf.safe_cell(0, 8, f"- {item}", ln=True)
            else:
                pdf.safe_multi_cell(0, 8, str(value))
            pdf.ln(3)
            
    elif isinstance(analysis, str):
        # Use enhanced text formatting
        pdf.add_analysis_text(analysis)
    else:
        pdf.set_font("Arial", "", 11)
        pdf.safe_cell(0, 8, "Analysis data format not supported", ln=True)
    
    # Add technical indicators with better formatting
    if active_indicators and not data.empty:
        pdf.set_font("Arial", "B", 14)
        pdf.safe_cell(0, 10, "Technical Indicators Summary", ln=True)
        pdf.ln(3)
        pdf.add_indicator_summary(data, active_indicators)
    
    # Add risk analysis with enhanced styling
    pdf.set_font("Arial", "B", 14)
    pdf.safe_cell(0, 10, "Risk Assessment & Market Levels", ln=True)
    pdf.ln(3)
    pdf.add_risk_analysis(data, levels, options_data)
    
    # Add disclaimer
    pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.safe_multi_cell(0, 6, 
        "DISCLAIMER: This analysis is generated by AI for educational purposes only. "
        "It should not be considered as financial advice. Always conduct your own research "
        "and consult with a qualified financial advisor before making any investment decisions. "
        "Past performance does not guarantee future results.")
    
    # Save the PDF
    try:
        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return False
