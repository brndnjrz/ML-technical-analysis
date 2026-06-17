from typing import Dict, Any
from ..trading_strategies import strategies_data


class BaseAgent:
    """Shared base for all AI agents — provides config and strategies_db."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategies_db = {s['Strategy']: s for s in strategies_data}
