# FreqAI Optimization

Advanced feature generation and risk management tools for FreqTrade's FreqAI module.

## Features

- **Advanced Feature Generator**: Comprehensive technical indicator generation using the `ta` library
  - Trend indicators (SMA, EMA, MACD, ADX)
  - Momentum indicators (RSI, Stochastic, CCI)
  - Volatility indicators (Bollinger Bands, ATR)
  - Volume indicators (OBV, VWAP)
  - Custom features (Price momentum, Volatility)
  - Feature normalization

- **Portfolio Risk Manager**: Sophisticated risk management and position sizing
  - Kelly Criterion-based position sizing
  - Volatility-adjusted position sizes
  - Maximum drawdown monitoring
  - Portfolio risk limits
  - Performance metrics calculation

## Installation

1. Create a Python 3.11 virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. Install the package and dependencies:
```bash
pip install -e .
```

3. Install FreqTrade UI:
```bash
freqtrade install-ui
```

## Running the Bot

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Start the bot:
```bash
freqtrade trade \
    --config config/freqtrade_config.json \
    --strategy-path strategy \
    --freqaimodel LightGBMRegressorMultiTarget
```

3. Access the web interface:
- Open your browser and go to: `http://127.0.0.1:8080`
- Login with the default credentials:
  - Username: freqai
  - Password: optimization

### Bot Configuration

The bot runs with the following settings:
- Dry run mode enabled (paper trading)
- Initial capital: 10,000 USDT
- Maximum position size: 10% of portfolio
- Maximum risk per trade: 2%
- Target Sharpe ratio: 2.0
- Maximum drawdown threshold: 20%

Trading pairs:
- BTC/USDT, ETH/USDT, BNB/USDT
- ADA/USDT, SOL/USDT, XRP/USDT
- DOT/USDT, DOGE/USDT, AVAX/USDT

### Monitoring the Bot

1. View real-time logs:
```bash
tail -f freqai_strategy.log
```

2. Check the database for trades:
```bash
sqlite3 tradesv3.dryrun.sqlite
```

3. Monitor via Web UI:
- Performance metrics
- Open trades
- Trade history
- Strategy parameters
- FreqAI model status

## Usage Examples

### Feature Generation

```python
from freqai_optimization import AdvancedFeatureGenerator

# Initialize feature generator
generator = AdvancedFeatureGenerator(lookback_window=20)

# Generate features from OHLCV data
features_df = generator.generate_features(ohlcv_df)

# Normalize features
normalized_df = generator.normalize_features(features_df)
```

### Risk Management

```python
from freqai_optimization import PortfolioRiskManager

# Initialize risk manager
risk_manager = PortfolioRiskManager(
    max_position_size=0.1,
    max_total_risk=0.02,
    risk_free_rate=0.02,
    target_sharpe=2.0,
    max_drawdown_threshold=0.2
)

# Calculate position size
position_size = risk_manager.calculate_position_size(
    symbol="BTC/USD",
    entry_price=50000,
    stop_loss=49000,
    confidence=0.75,
    volatility=0.2
)

# Update portfolio metrics
risk_manager.update_portfolio_metrics(
    current_value=100000,
    positions={"BTC/USD": 0.1}
)

# Check risk limits
is_safe, message = risk_manager.check_risk_limits()

# Calculate portfolio metrics
metrics = risk_manager.calculate_portfolio_metrics(returns_series)
```

## Requirements

- Python >= 3.11
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.0.0
- ta >= 0.10.0
- lightgbm >= 3.3.0
- mlflow >= 2.0.0
- optuna >= 3.0.0
- ccxt >= 4.0.0
- freqtrade >= 2024.1

## License

This project is licensed under the MIT License.
