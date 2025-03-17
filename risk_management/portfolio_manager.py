import numpy as np
import pandas as pd


class PortfolioRiskManager:

    def __init__(self, initial_capital=10000, max_risk_per_trade=0.02):
        """
        Initialize portfolio risk management

        :param initial_capital: Starting portfolio value
        :param max_risk_per_trade: Maximum risk allowed per trade (2% default)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = 0.1  # 10% maximum portfolio drawdown

        # Trade tracking
        self.open_trades = []
        self.trade_history = []

    def calculate_position_size(self, model_prediction=None, market_volatility=None):
        """
        Dynamically calculate optimal position size

        :param model_prediction: Confidence of trade prediction
        :param market_volatility: Current market volatility
        :return: Position size in dollars
        """
        # Base position sizing
        base_position = self.current_capital * self.max_risk_per_trade

        # Adjust based on model confidence if provided
        if model_prediction is not None:
            confidence_multiplier = abs(model_prediction)
            base_position *= confidence_multiplier

        # Adjust based on market volatility if provided
        if market_volatility is not None:
            volatility_adjustment = 1 / (1 + market_volatility)
            base_position *= volatility_adjustment

        # Ensure position size is within safe limits
        max_position = self.current_capital * 0.2  # No more than 20% in single trade
        return min(base_position, max_position)

    def simulate_trades(self, predictions):
        """
        Simulate trades based on model predictions

        Parameters:
        -----------
        predictions : dict or pandas.DataFrame
            Model predictions containing at least asset/symbol identifiers and predicted values

        Returns:
        --------
        dict
            Simulation results including portfolio performance and trade metrics
        """
        if isinstance(predictions, dict):
            predictions = pd.DataFrame.from_dict(predictions)

        # Initialize simulation variables
        starting_value = self.portfolio_value
        current_value = starting_value
        max_value = starting_value
        trades = []

        # Process each prediction
        for index, row in predictions.iterrows():
            # Extract relevant data (adjust column names as needed)
            symbol = row.get('symbol', f'Asset_{index}')
            prediction = row.get('prediction', 0)
            confidence = row.get('confidence', 0.5)
            price = row.get('price', 100)  # Default price if not provided

            # Skip if prediction or confidence is too low
            if abs(prediction) < 0.01 or confidence < 0.3:
                continue

            # Determine trade direction (buy/sell)
            direction = 1 if prediction > 0 else -1

            # Calculate position size
            position_size = self.calculate_position_size(
                prediction, confidence)
            shares = position_size / price

            # Record trade
            trade = {
                'timestamp': row.get('timestamp', index),
                'symbol': symbol,
                'direction': 'buy' if direction > 0 else 'sell',
                'price': price,
                'shares': shares,
                'position_value': position_size,
                'prediction': prediction,
                'confidence': confidence
            }
            trades.append(trade)

            # Update portfolio value (simplified simulation)
            # In a real simulation, you would track these positions and close them based on criteria
            simulated_return = prediction * confidence * 0.1  # Simplified return calculation
            trade_pnl = position_size * simulated_return
            current_value += trade_pnl

            # Update maximum portfolio value for drawdown calculation
            max_value = max(max_value, current_value)

        # Calculate performance metrics
        total_return = (current_value - starting_value) / starting_value
        max_drawdown = (max_value - current_value) / \
            max_value if max_value > current_value else 0

        # Compile results
        results = {
            'starting_value': starting_value,
            'ending_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades),
            'trades': trades
        }

        # Store trade history
        self.trade_history.extend(trades)

        return results

    def assess_risk(self, portfolio, predictions):
        """
        Assess portfolio risk based on current positions and predictions
        """
        # Calculate overall portfolio risk
        total_exposure = sum(abs(pos['value']) for pos in portfolio.values())
        exposure_ratio = total_exposure / self.portfolio_value

        # Calculate concentration risk
        max_position = max([abs(pos['value'])
                           for pos in portfolio.values()], default=0)
        concentration_risk = max_position / \
            self.portfolio_value if self.portfolio_value > 0 else 0

        # Calculate correlation risk (simplified)
        correlation_risk = 0.5  # Placeholder for actual correlation calculation

        # Combine risk factors
        overall_risk = (exposure_ratio * 0.4 +
                        concentration_risk * 0.3 +
                        correlation_risk * 0.3)

        return {
            'overall_risk': overall_risk,
            'exposure_ratio': exposure_ratio,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk
        }

    def generate_risk_report(self):
        """
        Generate a comprehensive risk report
        """
        # Analyze trade history
        if not self.trade_history:
            return {"error": "No trade history available"}

        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(self.trade_history)

        # Calculate key risk metrics
        win_rate = len(trades_df[trades_df['prediction'] > 0]) / len(trades_df)
        avg_position_size = trades_df['position_value'].mean()
        max_position_size = trades_df['position_value'].max()

        # Generate report
        report = {
            'portfolio_value': self.portfolio_value,
            'trade_count': len(self.trade_history),
            'win_rate': win_rate,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size,
            'risk_params': self.risk_params
        }

        return report
