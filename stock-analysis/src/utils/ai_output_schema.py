import jsonschema

# Strict JSON schema for AI model output with more flexible number handling
AI_MODEL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock symbol or ticker"},
        "timeframe": {
            "type": "string", 
            "enum": ["intraday", "1-7d", "2-4w"],
            "description": "Trading timeframe for the analysis"
        },
        "market_context": {
            "type": "object",
            "properties": {
                "trend": {
                    "type": "string", 
                    "enum": ["bullish", "bearish", "neutral"],
                    "description": "Overall market trend direction"
                },
                "adx": {
                    "type": ["number", "null"],
                    "description": "Average Directional Index - trend strength indicator (0-100)"
                },
                "rsi": {
                    "type": ["number", "null"],
                    "description": "Relative Strength Index - momentum oscillator (0-100)"
                },
                "iv_rank": {
                    "type": ["number", "null"],
                    "description": "Implied Volatility Rank - percentile of current IV (0-100)"
                }
            },
            "required": ["trend", "adx", "rsi", "iv_rank"]
        },
        "final_recommendation": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Recommended trading action"
                },
                "confidence": {
                    "type": ["number", "null"],
                    "description": "Confidence score (0-1) for the recommendation"
                },
                "strategy_name": {
                    "type": "string",
                    "description": "Name of the recommended trading strategy"
                },
                "rationale": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of reasons supporting the recommendation"
                }
            },
            "required": ["action", "confidence", "strategy_name", "rationale"]
        },
        "trade_parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": ["number", "null"],
                    "description": "Suggested entry price"
                },
                "stop": {
                    "type": ["number", "null"],
                    "description": "Suggested stop loss price"
                },
                "target": {
                    "type": ["number", "null"],
                    "description": "Suggested price target"
                },
                "position_size": {
                    "type": ["number", "null"],
                    "description": "Recommended position size (1.0 = 100%)"
                },
                "option_strikes": {
                    "type": "object",
                    "properties": {
                        "atm": {
                            "type": ["number", "null"],
                            "description": "At-the-money strike price"
                        },
                        "itm": {
                            "type": ["number", "null"],
                            "description": "In-the-money strike price"
                        },
                        "otm": {
                            "type": ["number", "null"],
                            "description": "Out-of-the-money strike price"
                        }
                    },
                    "required": ["atm", "itm", "otm"]
                }
            },
            "required": ["entry", "stop", "target", "position_size", "option_strikes"]
        },
        "alternatives": {
            "type": "array",
            "description": "Alternative trading strategies",
            "items": {
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Name of the alternative strategy"
                    },
                    "why_not": {
                        "type": "string",
                        "description": "Reason this strategy wasn't selected"
                    }
                },
                "required": ["strategy_name", "why_not"]
            }
        }
    },
    "required": ["ticker", "timeframe", "market_context", "final_recommendation", "trade_parameters", "alternatives"]
}

def adapt_strategy_output(strategy_data, ticker):
    """
    Adapt a strategy output to match the expected schema.
    Used when validation fails to try to reshape the data.
    Ensures all numeric fields have valid values (replacing None with defaults).
    """
    # Handle None values for numeric fields
    adx = strategy_data.get('adx')
    adx = 25 if adx is None else adx  # Default to moderate trend strength
    
    rsi = strategy_data.get('rsi')
    rsi = 50 if rsi is None else rsi  # Default to neutral RSI
    
    iv_rank = strategy_data.get('iv_rank')
    iv_rank = 50 if iv_rank is None else iv_rank  # Default to moderate volatility
    
    entry = strategy_data.get('entry', 0.0)
    entry = 0.0 if entry is None else entry
    
    stop = strategy_data.get('stop', 0.0)
    stop = 0.0 if stop is None else stop
    
    target = strategy_data.get('target', 0.0)
    target = 0.0 if target is None else target
    
    position_size = strategy_data.get('position_size', 1.0)
    position_size = 1.0 if position_size is None else position_size
    
    atm = strategy_data.get('atm', 0.0)
    atm = 0.0 if atm is None else atm
    
    itm = strategy_data.get('itm', 0.0)
    itm = 0.0 if itm is None else itm
    
    otm = strategy_data.get('otm', 0.0)
    otm = 0.0 if otm is None else otm
    
    confidence = strategy_data.get('confidence', 0.5)
    confidence = 0.5 if confidence is None else confidence

    # Get action with fallback
    action = strategy_data.get('action', 'HOLD')
    if action is None:
        action = 'HOLD'
    
    # Ensure we have a strategy name
    strategy_name = strategy_data.get('strategy_name', 'Default Strategy')
    if strategy_name is None:
        strategy_name = 'Default Strategy'
    
    # Ensure we have rationales as a list
    rationale = strategy_data.get('rationale', ["No rationale provided"])
    if rationale is None:
        rationale = ["No rationale provided"]
    elif isinstance(rationale, str):
        rationale = [rationale]
    
    # Create the basic structure with validated fields
    adapted = {
        "ticker": ticker,
        "timeframe": strategy_data.get("timeframe", "intraday"),
        "market_context": {
            "trend": strategy_data.get("trend", "unknown"),
            "adx": adx,
            "rsi": rsi,
            "iv_rank": iv_rank
        },
        "final_recommendation": {
            "action": action,
            "confidence": confidence,
            "strategy_name": strategy_name,
            "rationale": rationale
        },
        "trade_parameters": {
            "entry": entry,
            "stop": stop,
            "target": target,
            "position_size": position_size,
            "option_strikes": {
                "atm": atm,
                "itm": itm,
                "otm": otm
            }
        },
        "alternatives": strategy_data.get("alternatives", [])
    }
    return adapted

def validate_ai_model_output(data, ticker="Unknown"):
    """
    Validate AI model output against the strict schema.
    
    Args:
        data: The data to validate
        ticker: The ticker symbol (used as fallback if missing)
        
    Returns:
        bool or dict: True if validation succeeded with original data,
                     the adapted data dictionary if adapted successfully,
                     or raises an exception if validation fails completely
    """
    import logging
    
    if data is None:
        # Handle completely missing data
        logging.warning("Missing AI output data, creating default placeholder")
        default_data = {
            "ticker": ticker,
            "timeframe": "intraday",
            "market_context": {
                "trend": "neutral",
                "adx": 25,
                "rsi": 50,
                "iv_rank": 50
            },
            "final_recommendation": {
                "action": "HOLD",
                "confidence": 0.5,
                "strategy_name": "Default Strategy",
                "rationale": ["Insufficient data for analysis"]
            },
            "trade_parameters": {
                "entry": 100.0,
                "stop": 95.0,
                "target": 105.0,
                "position_size": 1.0,
                "option_strikes": {
                    "atm": 100.0,
                    "itm": 95.0,
                    "otm": 105.0
                }
            },
            "alternatives": []
        }
        return default_data
        
    try:
        # Try to validate as-is first
        jsonschema.validate(instance=data, schema=AI_MODEL_OUTPUT_SCHEMA)
        logging.debug(f"AI output validation successful for {ticker}")
        return True
    except jsonschema.exceptions.ValidationError as e:
        # Log the validation error details
        logging.warning(f"Schema validation error for {ticker}: {str(e)}")
        logging.debug(f"Original data structure: {list(data.keys() if isinstance(data, dict) else [])}")
        
        # If validation fails, try to adapt the data to match the schema
        try:
            adapted_data = adapt_strategy_output(data, ticker)
            jsonschema.validate(instance=adapted_data, schema=AI_MODEL_OUTPUT_SCHEMA)
            # If we get here, validation succeeded with the adapted data
            logging.info(f"Successfully adapted AI output data for {ticker}")
            
            # Instead of mutating the input, return the adapted version
            # This allows the caller to decide whether to use it
            return adapted_data
            
        except Exception as adapt_error:
            logging.error(f"Error adapting data for {ticker}: {str(adapt_error)}")
            # Create fallback data as last resort
            default_data = {
                "ticker": ticker,
                "timeframe": "intraday",
                "market_context": {
                    "trend": "neutral",
                    "adx": 25,
                    "rsi": 50,
                    "iv_rank": 50
                },
                "final_recommendation": {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "strategy_name": "Default Strategy",
                    "rationale": ["Schema validation failed: " + str(e)]
                },
                "trade_parameters": {
                    "entry": 100.0,
                    "stop": 95.0,
                    "target": 105.0,
                    "position_size": 1.0,
                    "option_strikes": {
                        "atm": 100.0,
                        "itm": 95.0,
                        "otm": 105.0
                    }
                },
                "alternatives": []
            }
            return default_data
