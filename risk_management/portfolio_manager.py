import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple


class PortfolioRiskManager:
    """
    Portfolio risk manager for optimizing position sizes and managing risk
    in trading strategies.
    """

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_total_risk: float = 0.02,
        risk_free_rate: float = 0.02,
        target_sharpe: float = 2.0,
        max_drawdown_threshold: float = 0.2
    ):
        """
        Initialize the portfolio risk manager.
        
        Args:
            max_position_size (float): Maximum position size as fraction of portfolio.
            max_total_risk (float): Maximum total portfolio risk as fraction.
            risk_free_rate (float): Annual risk-free rate for Sharpe calculation.
            target_sharpe (float): Target Sharpe ratio for position sizing.
            max_drawdown_threshold (float): Maximum allowable drawdown threshold.
        """
        self.max_position_size = max_position_size
        self.max_total_risk = max_total_risk
        self.risk_free_rate = risk_free_rate
        self.target_sharpe = target_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        
        # Initialize portfolio metrics
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = 0.0
        self.drawdown_history: list = []
        
    def update_capital(self, new_capital: float) -> None:
        """
        Update the current capital.
        
        Args:
            new_capital (float): New capital amount
        """
        self.portfolio_value = new_capital
        
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: float
    ) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol (str): Trading pair symbol.
            entry_price (float): Entry price for the position.
            stop_loss (float): Stop loss price.
            confidence (float): Model confidence score (0-1).
            volatility (float): Current market volatility.
            
        Returns:
            float: Recommended position size as fraction of portfolio.
        """
        # Calculate risk per trade
        risk_per_trade = abs(entry_price - stop_loss) / entry_price
        
        # Adjust position size based on volatility
        volatility_factor = 1.0 / (1.0 + volatility)
        
        # Calculate Kelly criterion
        win_prob = 0.5 + (confidence - 0.5) * 2  # Scale confidence to win probability
        loss_prob = 1 - win_prob
        kelly_fraction = (win_prob - loss_prob) / risk_per_trade
        
        # Apply constraints and adjustments
        position_size = min(
            kelly_fraction * volatility_factor,
            self.max_position_size
        )
        
        # Ensure position respects total portfolio risk
        total_risk = position_size * risk_per_trade
        if total_risk > self.max_total_risk:
            position_size = self.max_total_risk / risk_per_trade
        
        return max(0.0, position_size)
    
    def update_portfolio_metrics(
        self,
        current_value: float,
        positions: Dict[str, float]
    ) -> None:
        """
        Update portfolio metrics and track drawdown.
        
        Args:
            current_value (float): Current portfolio value.
            positions (Dict[str, float]): Current positions and their sizes.
        """
        self.portfolio_value = current_value
        self.current_positions = positions.copy()
        
        # Calculate drawdown
        if self.drawdown_history:
            peak = max(self.drawdown_history)
            drawdown = (peak - current_value) / peak
            self.drawdown_history.append(drawdown)
        else:
            self.drawdown_history.append(0.0)
    
    def check_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if current portfolio state respects risk limits.
        
        Returns:
            Tuple[bool, str]: (is_safe, message) indicating if portfolio is within risk limits.
        """
        # Check total position exposure
        total_exposure = sum(abs(pos) for pos in self.current_positions.values())
        if total_exposure > 1.0:
            return False, "Total position exposure exceeds 100%"
        
        # Check drawdown limit
        current_drawdown = self.drawdown_history[-1] if self.drawdown_history else 0.0
        if current_drawdown > self.max_drawdown_threshold:
            return False, f"Maximum drawdown threshold exceeded: {current_drawdown:.2%}"
        
        # Check individual position sizes
        for symbol, size in self.current_positions.items():
            if abs(size) > self.max_position_size:
                return False, f"Position size for {symbol} exceeds maximum allowed"
        
        return True, "Portfolio within risk limits"
    
    def calculate_portfolio_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> Dict[str, float]:
        """
        Calculate key portfolio metrics.
        
        Args:
            returns (pd.Series): Historical returns series.
            window (int): Rolling window size for calculations.
            
        Returns:
            Dict[str, float]: Dictionary of portfolio metrics.
        """
        # Calculate rolling metrics
        rolling_std = returns.rolling(window=window).std()
        rolling_mean = returns.rolling(window=window).mean()
        
        # Annualize metrics
        annual_return = rolling_mean.iloc[-1] * 252
        annual_volatility = rolling_std.iloc[-1] * np.sqrt(252)
        
        # Calculate Sharpe ratio
        excess_returns = annual_return - self.risk_free_rate
        sharpe_ratio = excess_returns / annual_volatility if annual_volatility != 0 else 0.0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': drawdowns.iloc[-1]
        }

    def add_position(self, symbol: str, amount: float, entry_price: float,
                    stop_loss: Optional[float] = None) -> bool:
        """
        Add a new position to track.
        
        Args:
            symbol (str): Trading pair symbol
            amount (float): Position size in base currency
            entry_price (float): Entry price
            stop_loss (float, optional): Stop loss price
            
        Returns:
            bool: True if position was added successfully
        """
        try:
            position_risk = (amount * entry_price) / self.portfolio_value
            
            if stop_loss:
                risk_amount = abs(entry_price - stop_loss) * amount
                position_risk = risk_amount / self.portfolio_value
            
            if position_risk > self.max_total_risk:
                return False
            
            self.current_positions[symbol] = amount
            
            return True
            
        except Exception as e:
            print(f"Error adding position: {e}")
            return False
            
    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from tracking.
        
        Args:
            symbol (str): Trading pair symbol
        """
        try:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
                
        except Exception as e:
            print(f"Error removing position: {e}")
            
    def adjust_position_risk(self, symbol: str, new_stop_loss: float) -> None:
        """
        Adjust the risk of an existing position.
        
        Args:
            symbol (str): Trading pair symbol
            new_stop_loss (float): New stop loss price
        """
        try:
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                risk_amount = abs(position - new_stop_loss) * position
                self.current_positions[symbol] = risk_amount / self.portfolio_value
                
        except Exception as e:
            print(f"Error adjusting position risk: {e}")
            
    def get_position_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific position.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict[str, Any]: Position information or None if not found
        """
        return self.current_positions.get(symbol)

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
                symbol, price, price - price * 0.01, confidence, 0.0)
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
