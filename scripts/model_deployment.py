import joblib
import numpy as np
import pandas as pd
from models.trading_model import TradingModel
from data.preprocessors.feature_generator import FeatureGenerator

def save_best_model(study, model_class):
    """
    Save the best model based on optimization results
    
    :param study: Optuna study object
    :param model_class: Model class to instantiate
    """
    # Extract best hyperparameters
    best_params = study.best_params
    
    # Create model with best parameters
    final_model = model_class(best_params)
    
    # Simulate data preparation (replace with actual data collection)
    np.random.seed(42)
    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(500)),
        'open': np.random.randn(500),
        'high': np.random.randn(500),
        'low': np.random.randn(500)
    })
    
    # Generate features
    features_df = FeatureGenerator.generate_features(df)
    
    # Prepare data
    features = features_df.drop('close', axis=1)
    target = features_df['close']
    
    # Prepare and train the model
    X_train, X_test, y_train, y_test = final_model.prepare_data(features, target)
    final_model.train(X_train, y_train)
    
    # Evaluate the model
    evaluation_metrics = final_model.evaluate(X_test, y_test)
    
    # Save the model
    joblib.dump(final_model.model, 'best_trading_model.joblib')
    
    print("Best model saved successfully!")
    print("Model Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")
