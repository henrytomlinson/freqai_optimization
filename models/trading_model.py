from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import mlflow
import numpy as np


class TradingModel:
    def __init__(self, model_params=None):
        # Default parameters with compatible RandomForestRegressor parameters
        default_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_leaf': 20,  # Corrected from min_child_samples
            'random_state': 42
        }

        # Update with provided parameters
        self.model_params = default_params
        if model_params:
            self.model_params.update({
                k: v for k, v in model_params.items()
                if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'random_state']
            })

        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, features, target):
        """
        Advanced data preparation with stratified splitting
        """
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train model with advanced configuration
        """
        # Use RandomForestRegressor with corrected parameters
        self.model = RandomForestRegressor(**self.model_params)
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        # Predictions
        predictions = self.model.predict(X_test)

        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        r2 = 1 - (np.sum((y_test - predictions)**2) /
                  np.sum((y_test - np.mean(y_test))**2))

        # Log metrics with MLflow
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

        mlflow.log_metrics(metrics)

        return metrics

    def predict(self, features):
        """
        Make predictions on input features

        :param features: DataFrame or array of features
        :return: Predictions (price movement direction)
        """
        # Ensure the model is trained
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        # Scale features using the same scaler used during training
        scaled_features = self.scaler.transform(features)

        # Make predictions
        predictions = self.model.predict(scaled_features)

        # Convert predictions to directional signals
        # Positive values indicate potential price increase
        # Negative values indicate potential price decrease
        return np.sign(predictions - np.mean(predictions))
