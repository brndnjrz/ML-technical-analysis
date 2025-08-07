from fpdf import FPDF
from PIL import Image
import pandas as pd
import os


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
            self.cell(0, 10, "Chart could not be added to the report", ln=True)

    def add_analysis_text(self, text):
        text = text.replace("*", "").replace("#", "")
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
        self.ln(2)

    def add_indicator_summary(self, data, active_indicators):
        self.set_font("Arial", "B", 13)
        self.cell(0, 10, "Technical Indicator Summary", ln=True)
        self.set_font("Arial", "", 11)

        # Map indicator names to their corresponding DataFrame column names
        indicator_mapping = {
            # Regular Timeframe Indicators
            ## RSI Variations
            "RSI (14)": "RSI_14",
            "RSI (21)": "RSI_21",
            "RSI (Standard)": "RSI",
            
            ## Moving Averages
            "SMA (20)": "SMA_20",
            "SMA (50)": "SMA_50",
            "EMA (20)": "EMA_20",
            "EMA (50)": "EMA_50",
            
            ## Volatility
            "ATR": "ATR",
            "Bollinger Bands": ["BB_upper", "BB_middle", "BB_lower"],
            
            ## Momentum
            "MACD": ["MACD", "MACD_Signal", "MACD_Hist"],
            "MACD Line": "MACD_Line",
            "Signal Line": "Signal_Line",
            "MACD Histogram": "MACD_Histogram",
            
            ## Volume
            "OBV": "OBV",
            "Volume": "Volume",
            "VWAP": "VWAP",
            
            ## Trend
            "ADX": "ADX",

            # 5-Minute Timeframe Indicators
            ## RSI
            "RSI (5min)": "RSI_5m",
            
            ## Moving Averages
            "SMA (20) 5min": "SMA_20_5m",
            "SMA (50) 5min": "SMA_50_5m",
            "EMA (20) 5min": "EMA_20_5m",
            "EMA (50) 5min": "EMA_50_5m",
            
            ## Volatility
            "ATR 5min": "ATR_5m",
            "Bollinger Bands 5min": ["BB_upper_5m", "BB_middle_5m", "BB_lower_5m"],
            
            ## Momentum
            "MACD 5min": ["MACD_5m", "MACD_Signal_5m", "MACD_Hist_5m"],
            "Stochastic Fast 5min": ["STOCH_%K_5m", "STOCH_%D_5m"],
            
            ## Volume
            "OBV 5min": "OBV_5m",
            "VWAP 5min": "VWAP_5m",
            
            ## Trend
            "ADX 5min": "ADX_5m",
            
            # Predictions and Analysis
            "Predicted Close": "Predicted_Close",
            "Predicted Price Change": "Predicted_Price_Change",
            "Volatility": "volatility",
            "Returns": "returns",
            
            # 5-Minute Timeframe Indicators
            ## RSI
            "RSI (5min)": "RSI_5m",
            
            ## Moving Averages
            "SMA (20) 5min": "SMA_20_5m",
            "SMA (50) 5min": "SMA_50_5m",
            "EMA (20) 5min": "EMA_20_5m",
            "EMA (50) 5min": "EMA_50_5m",
            
            ## Volatility
            "ATR 5min": "ATR_5m",
            "Bollinger Bands 5min": ["BB_upper_5m", "BB_middle_5m", "BB_lower_5m"],
            
            ## Momentum
            "MACD 5min": ["MACD_5m", "MACD_Signal_5m", "MACD_Hist_5m"],
            "Stochastic Fast 5min": ["STOCH_%K_5m", "STOCH_%D_5m"],
            
            ## Volume
            "OBV 5min": "OBV_5m",
            "VWAP 5min": "VWAP_5m",
            
            ## Trend
            "ADX 5min": "ADX_5m",
            
            # Price Prediction
            "Predicted Close": "Predicted_Close",
            "Predicted Price Change": "Predicted_Price_Change",
            
            # Additional Metrics
            "Volatility": "volatility",
            "Returns": "returns",
            
            # Trend Indicators
            "ADX": "ADX",
            "VWAP": "VWAP"
        }

        # Print available columns for debugging
        print("Available columns in data:", sorted(data.columns))
        
        for ind in active_indicators:
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
                                values.append(f"{col.split('_')[-1]}: {val:.4f}")
                    val_str = " | ".join(values) if values else "N/A"
                    self.cell(0, 8, f"{ind}: {val_str}", ln=True)
                else:
                    # Handle single-value indicators
                    if col_names in data.columns:
                        val = data[col_names].iloc[-1]
                        if not pd.isna(val):
                            # Format based on indicator type
                            if any(x in ind for x in ["RSI", "ADX", "Stochastic"]):
                                val_str = f"{val:.2f}"  # Percentage-style indicators
                            elif any(x in ind for x in ["ATR", "SMA", "EMA", "VWAP", "Predicted Close"]):
                                val_str = f"${val:.2f}"  # Price-based indicators
                            elif "OBV" in ind:
                                val_str = f"{val:,.0f}"  # Volume-based indicators
                            elif "Volatility" in ind:
                                val_str = f"{val:.2%}"  # Percentage format for volatility
                            elif "Returns" in ind:
                                val_str = f"{val:.2%}"  # Percentage format for returns
                            elif "Predicted Price Change" in ind:
                                val_str = f"${val:.2f}"  # Price change format
                            else:
                                val_str = f"{val:.4f}"  # Default format for other indicators
                        else:
                            val_str = "N/A"
                    else:
                        val_str = "N/A"
                    self.cell(0, 8, f"{ind}: {val_str}", ln=True)
            except Exception as e:
                print(f"Error processing indicator {ind}: {str(e)}")
                self.cell(0, 8, f"{ind}: Error", ln=True)
        
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
