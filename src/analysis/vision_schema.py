"""
Schema validation and structured output for AI vision analysis
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import json
import re
from enum import Enum
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class VisionAnalysisSchema(BaseModel):
    """Structured schema for vision AI analysis output"""
    
    trend: TrendDirection = Field(
        description="Overall trend direction from chart analysis"
    )
    
    support: List[float] = Field(
        default_factory=list,
        description="Support levels identified from chart (sorted ascending)"
    )
    
    resistance: List[float] = Field(
        default_factory=list,
        description="Resistance levels identified from chart (sorted ascending)"
    )
    
    entries: Dict[str, Optional[float]] = Field(
        default_factory=lambda: {"buy_above": None, "sell_below": None},
        description="Entry trigger levels"
    )
    
    risk: RiskLevel = Field(
        description="Risk assessment based on volatility and market conditions"
    )
    
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the analysis (0-1)"
    )
    
    pattern_recognition: List[str] = Field(
        default_factory=list,
        description="Identified chart patterns"
    )
    
    volume_analysis: Optional[str] = Field(
        None,
        description="Volume pattern assessment"
    )
    
    @field_validator('support', 'resistance')
    @classmethod
    def validate_price_levels(cls, v):
        """Ensure price levels are reasonable"""
        if not v:
            return v
            
        # Remove any levels that are clearly outliers (>50% away from reasonable range)
        # This is a simple validation - in production you'd use current price context
        if len(v) > 1:
            median_level = sorted(v)[len(v) // 2]
            filtered_levels = [
                level for level in v 
                if abs(level - median_level) / median_level < 0.5
            ]
            if len(filtered_levels) < len(v):
                field_name = 'price_levels'  # Generic name since we can't access field info easily in V2
                logger.warning(f"Filtered {len(v) - len(filtered_levels)} outlier {field_name} levels")
            return sorted(filtered_levels)
        
        return sorted(v)
    
    @field_validator('entries')
    @classmethod
    def validate_entries(cls, v):
        """Ensure entry levels are reasonable"""
        if v.get('buy_above') and v.get('sell_below'):
            if v['buy_above'] <= v['sell_below']:
                logger.warning("Buy level should be above sell level, correcting...")
                # Swap or null out conflicting levels
                v['buy_above'] = None
                v['sell_below'] = None
        return v

class StructuredVisionParser:
    """Parse and validate vision model outputs into structured format"""
    
    def __init__(self):
        self.json_extraction_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
            r'\{[^}]+\}',  # Simple JSON
        ]
    
    def parse_vision_output(self, raw_output: str, current_price: float = 0) -> Dict[str, Any]:
        """
        Parse raw vision output into validated structured format
        
        Args:
            raw_output: Raw text from vision model
            current_price: Current stock price for validation
            
        Returns:
            Dict containing validated analysis or error info
        """
        try:
            # Try to extract JSON first
            structured_data = self._extract_json(raw_output)
            
            if structured_data:
                # Validate with Pydantic schema
                validated = self._validate_with_schema(structured_data, current_price)
                if validated:
                    return validated
            
            # Fallback: Parse from natural language
            logger.info("🔄 JSON extraction failed, parsing natural language...")
            fallback_data = self._parse_natural_language(raw_output, current_price)
            return self._validate_with_schema(fallback_data, current_price)
            
        except Exception as e:
            logger.error(f"❌ Vision parsing error: {e}")
            return self._create_fallback_analysis(current_price)
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text using multiple strategies"""
        try:
            # Clean the text first
            cleaned_text = text.strip()
            
            # Strategy 1: Try parsing entire text as JSON first (most common case)
            try:
                parsed = json.loads(cleaned_text)
                if isinstance(parsed, dict) and len(parsed) > 2:
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Find JSON block markers and extract content
            json_markers = ['```json', '```JSON', '```', 'JSON:', 'json:', '{']
            for marker in json_markers:
                if marker in cleaned_text:
                    if marker == '{':
                        # Find the first opening brace and try to extract complete JSON
                        start_idx = cleaned_text.find('{')
                        if start_idx >= 0:
                            # Find matching closing brace
                            brace_count = 0
                            end_idx = start_idx
                            for i in range(start_idx, len(cleaned_text)):
                                if cleaned_text[i] == '{':
                                    brace_count += 1
                                elif cleaned_text[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                            
                            if end_idx > start_idx:
                                json_candidate = cleaned_text[start_idx:end_idx]
                                try:
                                    parsed = json.loads(json_candidate)
                                    if isinstance(parsed, dict) and len(parsed) > 2:
                                        return parsed
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Extract content after marker
                        parts = cleaned_text.split(marker, 1)
                        if len(parts) > 1:
                            json_candidate = parts[1].split('```')[0].strip()
                            # Clean up common issues
                            json_candidate = json_candidate.replace('\n', ' ').replace('\r', '')
                            json_candidate = re.sub(r'\s+', ' ', json_candidate)  # Normalize whitespace
                            
                            try:
                                parsed = json.loads(json_candidate)
                                if isinstance(parsed, dict) and len(parsed) > 2:
                                    return parsed
                            except json.JSONDecodeError:
                                continue
            
            # Strategy 3: Enhanced regex patterns for JSON-like structures
            enhanced_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
                r'\{[^}]+\}',  # Simple JSON
                r'\{.*?"trend".*?\}',  # JSON containing our expected fields
                r'\{.*?"confidence".*?\}',
            ]
            
            for pattern in enhanced_patterns:
                matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    # Clean the match
                    match = match.strip()
                    # Fix common JSON formatting issues
                    match = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', match)  # Add quotes to keys
                    
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, dict) and len(parsed) > 2:
                            # Validate it has expected structure
                            if any(key in parsed for key in ['trend', 'confidence', 'support', 'resistance']):
                                return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Strategy 4: Log first 200 chars for debugging
            logger.debug(f"🔍 JSON extraction failed. First 200 chars: {cleaned_text[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"❌ JSON extraction error: {e}")
            return None
                
            return None
            
        except Exception as e:
            logger.error(f"❌ JSON extraction error: {e}")
            return None
    
    def _validate_with_schema(self, data: Dict[str, Any], current_price: float = 0) -> Dict[str, Any]:
        """Validate data against Pydantic schema"""
        try:
            # Apply price level validation if current price is available
            if current_price > 0:
                data = self._apply_price_bounds(data, current_price)
            
            # Create and validate schema
            validated = VisionAnalysisSchema(**data)
            
            # Convert back to dict with additional metadata
            result = validated.dict()
            result['schema_validation'] = 'passed'
            result['validation_timestamp'] = str(pd.Timestamp.now())
            
            logger.info(f"✅ Vision analysis validated: {result['trend']} trend, "
                       f"confidence {result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Schema validation failed: {e}")
            # Return data with validation failure note
            if isinstance(data, dict):
                data['schema_validation'] = 'failed'
                data['validation_error'] = str(e)
            return data if data else self._create_fallback_analysis(current_price)
    
    def _apply_price_bounds(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Apply reasonable bounds to price levels based on current price"""
        try:
            # Define reasonable bounds (±20% for normal analysis)
            lower_bound = current_price * 0.8
            upper_bound = current_price * 1.2
            
            # Filter support levels
            if 'support' in data and isinstance(data['support'], list):
                data['support'] = [
                    level for level in data['support']
                    if isinstance(level, (int, float)) and lower_bound <= level < current_price
                ]
            
            # Filter resistance levels  
            if 'resistance' in data and isinstance(data['resistance'], list):
                data['resistance'] = [
                    level for level in data['resistance'] 
                    if isinstance(level, (int, float)) and current_price < level <= upper_bound
                ]
            
            # Validate entry levels
            if 'entries' in data and isinstance(data['entries'], dict):
                entries = data['entries']
                if entries.get('buy_above') and not (current_price < entries['buy_above'] <= upper_bound):
                    entries['buy_above'] = None
                if entries.get('sell_below') and not (lower_bound <= entries['sell_below'] < current_price):
                    entries['sell_below'] = None
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Price bounds application error: {e}")
            return data
    
    def _parse_natural_language(self, text: str, current_price: float = 0) -> Dict[str, Any]:
        """Parse natural language vision output into structured format"""
        try:
            analysis = {
                'trend': 'neutral',
                'support': [],
                'resistance': [],
                'entries': {'buy_above': None, 'sell_below': None},
                'risk': 'medium',
                'confidence': 0.5,
                'pattern_recognition': [],
                'volume_analysis': None
            }
            
            text_lower = text.lower()
            
            # Extract trend
            if any(word in text_lower for word in ['bullish', 'uptrend', 'rising', 'bull']):
                analysis['trend'] = 'bullish'
            elif any(word in text_lower for word in ['bearish', 'downtrend', 'falling', 'bear']):
                analysis['trend'] = 'bearish'
            
            # Extract price levels using regex
            price_pattern = r'\$?(\d+\.?\d*)'
            prices = [float(match) for match in re.findall(price_pattern, text)]
            
            if current_price > 0 and prices:
                # Classify prices as support/resistance based on current price
                analysis['support'] = sorted([p for p in prices if p < current_price * 0.99])
                analysis['resistance'] = sorted([p for p in prices if p > current_price * 1.01])
            
            # Extract risk assessment
            if any(word in text_lower for word in ['high risk', 'volatile', 'risky']):
                analysis['risk'] = 'high'
            elif any(word in text_lower for word in ['low risk', 'stable', 'safe']):
                analysis['risk'] = 'low'
            
            # Extract confidence from language cues
            confidence_cues = {
                'very confident': 0.9,
                'confident': 0.8,
                'likely': 0.7,
                'possible': 0.6,
                'uncertain': 0.4,
                'unclear': 0.3
            }
            
            for cue, conf_value in confidence_cues.items():
                if cue in text_lower:
                    analysis['confidence'] = conf_value
                    break
            
            # Pattern recognition
            patterns = ['triangle', 'wedge', 'flag', 'pennant', 'channel', 'breakout', 'breakdown']
            analysis['pattern_recognition'] = [p for p in patterns if p in text_lower]
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Natural language parsing error: {e}")
            return self._create_fallback_analysis(current_price)
    
    def _create_fallback_analysis(self, current_price: float = 0) -> Dict[str, Any]:
        """Create safe fallback analysis when parsing fails"""
        return {
            'trend': 'neutral',
            'support': [current_price * 0.98] if current_price > 0 else [],
            'resistance': [current_price * 1.02] if current_price > 0 else [],
            'entries': {'buy_above': None, 'sell_below': None},
            'risk': 'medium',
            'confidence': 0.3,  # Low confidence for fallback
            'pattern_recognition': [],
            'volume_analysis': None,
            'schema_validation': 'fallback',
            'note': 'Fallback analysis due to parsing failure'
        }

# Global parser instance
vision_parser = StructuredVisionParser()

def parse_vision_analysis(raw_output: str, current_price: float = 0) -> Dict[str, Any]:
    """Convenience function for parsing vision analysis"""
    return vision_parser.parse_vision_output(raw_output, current_price)

def create_vision_prompt_template(ticker: str, timeframe: str, current_price: float, 
                                atr: float, rsi: float, iv_rank: float) -> str:
    """
    Create standardized vision analysis prompt template with improved JSON formatting
    """
    return f"""
IMPORTANT: Respond with ONLY the JSON object below, no additional text, no explanations, no markdown formatting.

Analyze this {timeframe} chart for {ticker} and return this exact JSON structure:

{{
    "trend": "bullish",
    "support": [123.45, 120.00],
    "resistance": [130.50, 135.75], 
    "entries": {{
        "buy_above": 125.00,
        "sell_below": null
    }},
    "risk": "medium",
    "confidence": 0.75,
    "pattern_recognition": ["ascending triangle", "volume breakout"],
    "volume_analysis": "increasing volume on breakout"
}}

Context for your analysis:
- Current Price: ${current_price:.2f}
- ATR (daily volatility): ${atr:.2f} ({atr/current_price*100:.1f}%)
- RSI: {rsi:.1f}
- IV Rank: {iv_rank:.1f}%

Strict Rules:
1. Return ONLY the JSON object - no text before or after
2. Use "bullish", "bearish", or "neutral" for trend
3. Support levels must be < ${current_price*0.99:.2f}
4. Resistance levels must be > ${current_price*1.01:.2f}  
5. No levels outside ${current_price*0.8:.2f} - ${current_price*1.2:.2f} range
6. Use null for uncertain values, not "null" string
7. Use "low", "medium", or "high" for risk
8. Confidence must be 0.1 to 0.9 decimal

JSON ONLY - NO OTHER TEXT
"""
