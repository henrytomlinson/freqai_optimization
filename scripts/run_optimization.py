# scripts/run_optimization.py
import os
import mlflow
import optuna
import numpy as np
import pandas as pd

from config.base_config import FreqAIConfig
from data.collectors.crypto_data_collector import CryptoDataCollector
from data.preprocessors.advanced_feature_generator import AdvancedFeatureGenerator
from models.trading_model import TradingModel


def prepare_training_data():
    """
    Comprehensive data preparation with advanced feature engineering
    """
    # 1. Data Collection
    collector = CryptoDataCollector()

    # Fetch data for multiple pairs and timeframes
    pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['1h', '4h', '1d']

    # Aggregate data from multiple sources
    combined_data = []
    for pair in pairs:
        for timeframe in timeframes:
            # Fetch historical data
            df = collector.fetch_historical_data(pair, timeframe, limit=1000)

            # Generate advanced features
            features_df = AdvancedFeatureGenerator.generate_comprehensive_features(
                df)

            # Add pair and timeframe information
            features_df['pair'] = pair
            features_df['timeframe'] = timeframe

            combined_data.append(features_df)

    # Combine all data
    full_dataset = pd.concat(combined_data)

    # Feature selection
    selected_features = AdvancedFeatureGenerator.select_top_features(
        full_dataset)

    # Prepare target variable (e.g., predicting next period's return)
    target = selected_features['returns']
    features = selected_features.drop(['returns', 'pair', 'timeframe'], axis=1)

    return features, target


def main():
    # Load configuration
    config = FreqAIConfig()

    # Set up MLflow tracking
    mlflow_dir = os.path.join(os.getcwd(), 'mlflow_tracking')
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")

    # Set experiment name
    experiment_name = config.get('optimization', {}).get(
        'study_name', 'Advanced Trading Strategy')
    mlflow.set_experiment(experiment_name)

    # Prepare data
    features, target = prepare_training_data()

    # Create Optuna study for hyperparameter optimization
    study = optuna.create_study(
        direction='minimize',  # Minimize prediction error
        study_name=experiment_name
    )

    def objective(trial):
        # Hyperparameter search space
        model_params = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                        'random_state': 42

                        }

        # Create model with suggested hyperparameters
        model = TradingModel(model_params)

        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(features, target)

        # Train model
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(model_params)

            # Train and evaluate
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Return MSE for optimization
            return metrics['mse']

    # Run optimization
    study.optimize(objective, n_trials=50)

    # Print and log best results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    main()
