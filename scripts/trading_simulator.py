from risk_management.portfolio_manager import PortfolioRiskManager
from data.collectors.crypto_data_collector import CryptoDataCollector
from models.trading_model import TradingModel
from data.preprocessors.advanced_feature_generator import AdvancedFeatureGenerator
import numpy as np
import pandas as pd
import logging
import argparse
import sys
import traceback
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_simulator.log')
    ]
)
logger = logging.getLogger('trading_simulator')

# Imports


class TradingSimulator:
    def __init__(self, data_path='data/market_data.csv', model_params=None, risk_params=None):
        """
        Initialize the trading simulator

        :param data_path: Path to market data CSV
        :param model_params: Parameters for the trading model
        :param risk_params: Parameters for risk management
        """
        self.data_path = data_path
        self.data = None

        # Initialize model
        self.model = TradingModel(model_params or {})

        # Initialize risk manager
        initial_capital = risk_params.get(
            'portfolio_size', 10000) if risk_params else 10000
        self.risk_manager = PortfolioRiskManager(
            initial_capital=initial_capital)

    def load_data(self):
        """Load and preprocess simulation data"""
        logger.info(f"Loading data from {self.data_path}")
        try:
            # Load the data
            self.data = pd.read_csv(self.data_path)

            # Use AdvancedFeatureGenerator to add features
            logger.info("Generating advanced features")
            self.data = AdvancedFeatureGenerator.generate_comprehensive_features(
                self.data)

            logger.info(f"Loaded {len(self.data)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def generate_predictions(self):
        """Generate model predictions for the loaded data"""
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return None

        logger.info("Generating predictions")

        # Select top features
        features = AdvancedFeatureGenerator.select_top_features(self.data)

        # Prepare target variable (next period's return)
        target = self.data['returns']

        # Train the model
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            features, target)
        self.model.train(X_train, y_train)

        # Generate predictions
        predictions = {
            'symbol': self.data['symbol'].values,
            'timestamp': self.data['timestamp'].values,
            'price': self.data['close'].values,
            'prediction': self.model.predict(features),
            'confidence': np.abs(self.model.predict(features))
        }

        logger.info(f"Generated {len(predictions['prediction'])} predictions")
        return predictions

    def simulate_trading_scenario(self):
        """Run a complete trading simulation"""
        # Load data if not already loaded
        if self.data is None:
            success = self.load_data()
            if not success:
                return {"error": "Failed to load data"}

        # Generate predictions
        predictions = self.generate_predictions()
        if predictions is None:
            return {"error": "Failed to generate predictions"}

        # Simulate trades
        logger.info("Simulating trades")

        # Track portfolio performance
        portfolio_value = self.risk_manager.initial_capital
        portfolio_values = [portfolio_value]
        trade_directions = []
        max_drawdown = 0
        peak_portfolio_value = portfolio_value

        # Ensure we don't go out of bounds
        for i in range(len(predictions['prediction']) - 1):
            # Determine trade signal
            signal = predictions['prediction'][i]
            confidence = predictions['confidence'][i]
            current_price = predictions['price'][i]
            next_price = predictions['price'][i+1]

            # Position sizing based on confidence and volatility
            position_size = self.risk_manager.calculate_position_size(
                model_prediction=confidence,
                market_volatility=self.data['atr'].values[i] if 'atr' in self.data.columns else 0.01
            )

            # Trading logic with more nuanced decision-making
            trade_profit = 0
            if signal > 0.5:  # Strong buy signal
                # Long position
                trade_profit = position_size * \
                    (next_price - current_price) / current_price
                trade_directions.append(1)
            elif signal < -0.5:  # Strong sell signal
                # Short position
                trade_profit = position_size * \
                    (current_price - next_price) / current_price
                trade_directions.append(-1)
            else:
                trade_directions.append(0)

            # Update portfolio value
            portfolio_value += trade_profit
            portfolio_values.append(portfolio_value)

            # Track maximum drawdown
            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value

            drawdown = (peak_portfolio_value - portfolio_value) / \
                peak_portfolio_value
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate performance metrics
        total_return = (portfolio_value - self.risk_manager.initial_capital) / \
            self.risk_manager.initial_capital * 100

        results = {
            'num_trades': len(trade_directions),
            'ending_value': portfolio_value,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown * 100,
            'portfolio_values': portfolio_values,
            'trade_directions': trade_directions
        }

        return results


def visualize_performance(results):
    """
    Create a visualization of the trading simulation results
    """
    # Portfolio Value Over Time
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title('Portfolio Value Over Time')
    plt.plot(results['portfolio_values'])
    plt.xlabel('Trading Periods')
    plt.ylabel('Portfolio Value')

    # Trade Directions
    plt.subplot(2, 1, 2)
    plt.title('Trade Directions')
    plt.plot(results['trade_directions'], marker='o')
    plt.xlabel('Trading Periods')
    plt.ylabel('Trade Direction')
    plt.yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])

    plt.tight_layout()
    plt.savefig('trading_simulation_performance.png')
    plt.close()


def main():
    try:
        logger.debug("Starting trading simulator")

        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Run trading simulation')
        parser.add_argument('--data', type=str, default='data/market_data.csv',
                            help='Path to market data CSV file')
        parser.add_argument('--max_position', type=float, default=0.05,
                            help='Maximum position size as fraction of portfolio')
        parser.add_argument('--portfolio_size', type=float, default=100000,
                            help='Starting portfolio size')

        args = parser.parse_args()
        logger.debug(f"Arguments: {args}")

        # Set up risk parameters
        risk_params = {
            'max_position_size': args.max_position,
            'portfolio_size': args.portfolio_size
        }
        logger.debug(f"Risk parameters: {risk_params}")

        # Create and run simulator
        simulator = TradingSimulator(args.data, risk_params=risk_params)
        logger.debug("Simulator created")

        results = simulator.simulate_trading_scenario()
        logger.debug("Simulation completed")

        # Display results
        if 'error' in results:
            logger.error(results['error'])
        else:
            logger.info(
                f"Simulation completed with {results['num_trades']} trades")
            logger.info(
                f"Final portfolio value: ${results['ending_value']:.2f}")
            logger.info(f"Total return: {results['total_return_pct']:.2f}%")
            logger.info(
                f"Maximum drawdown: {results['max_drawdown_pct']:.2f}%")

            # Visualize performance
            visualize_performance(results)

    except Exception as e:
        # Catch and log any unexpected errors
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        print(f"Error details: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
