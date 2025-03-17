from typing import Dict, Any

class BaseConfig:
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = config_dict or {}
    
    def get(self, key: str, default: Any = None):
        return self.config.get(key, default)
    
    def update(self, new_config: Dict[str, Any]):
        self.config.update(new_config)

# Specific FreqAI configuration
class FreqAIConfig(BaseConfig):
    DEFAULT_CONFIG = {
        'model': {
            'type': 'LightGBMRegressorMultiTarget',
            'n_estimators': 400,
            'learning_rate': 0.02,
            'max_depth': 6
        },
        'data': {
            'timeframes': ['5m', '15m', '1h'],
            'pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        },
        'optimization': {
            'max_trials': 100,
            'study_name': 'FreqAI Optimization'
        }
    }

    def __init__(self, custom_config: Dict[str, Any] = None):
        super().__init__(self.DEFAULT_CONFIG)
        if custom_config:
            self.update(custom_config)
