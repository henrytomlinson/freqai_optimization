# FreqAI Optimization

A comprehensive project for optimizing FreqAI trading strategies using MLflow tracking and hyperparameter tuning.

## Overview

This project provides a framework for optimizing FreqTrade's FreqAI module, incorporating:
- Automated hyperparameter optimization using Optuna
- MLflow experiment tracking
- Custom trading strategies
- Risk management tools
- Performance visualization

## Project Structure

```
freqai_optimization/
├── mlflow_tracking/      # MLflow experiment tracking data
├── models/              # Trading model implementations
├── notebooks/          # Jupyter notebooks for analysis
├── optimization/       # Optimization related code
├── risk_management/    # Risk management tools
├── scripts/           # Utility scripts
├── strategy/          # Trading strategies
├── tests/            # Test suite
├── user_data/        # FreqTrade user data
└── utils/           # Utility functions
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/henrytomlinson/freqai_optimization.git
cd freqai_optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install TA-Lib (if needed):
```bash
bash install_talib_complete.sh
```

## Usage

### Running the Bot

```bash
./run_bot.sh
```

### Running Optimization

```bash
python scripts/run_optimization.py
```

### Viewing Results

1. Start MLflow UI:
```bash
mlflow ui
```

2. View optimization results in your browser at `http://localhost:5000`

## Features

- **Automated Optimization**: Uses Optuna for hyperparameter tuning
- **Experiment Tracking**: MLflow integration for tracking experiments
- **Risk Management**: Custom portfolio management tools
- **Visualization**: Tools for analyzing trading performance
- **VM Support**: Scripts for setting up and running on virtual machines

## Documentation

- [VM Setup Instructions](vm_instructions.md)
- [TA-Lib Installation Guide](talib_fix_instructions.md)
- [Optimization Report](optimization_report.md)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
