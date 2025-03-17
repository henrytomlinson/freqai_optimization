import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OptimizedRegressionModel(BaseRegressionModel):
    """
    Custom FreqAI model that uses the optimized RandomForestRegressor
    from the codebase. This model integrates with the FreqAI framework
    and uses the optimized hyperparameters.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Train the model with optimized hyperparameters
        
        :param data_dictionary: Dict containing the training data
        :param dk: FreqaiDataKitchen object
        :return: Trained model object
        """
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        
        # Get model parameters from config
        model_params = self.model_training_parameters
        
        # Create and train the model
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 400),
            max_depth=model_params.get('max_depth', 6),
            min_samples_leaf=model_params.get('min_samples_leaf', 20),
            random_state=model_params.get('random_state', 42),
            n_jobs=-1  # Use all available cores
        )
        
        # Fit the model
        model.fit(X, y)
        
        # Save feature importance
        self.save_feature_importance(model, dk)
        
        return model

    def predict(self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs) -> pd.DataFrame:
        """
        Make predictions with the trained model
        
        :param unfiltered_df: DataFrame with unfiltered data
        :param dk: FreqaiDataKitchen object
        :return: DataFrame with predictions
        """
        # Get the model
        model = self.model
        
        # Get the features
        features = dk.get_features_by_index(unfiltered_df)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Add predictions to the DataFrame
        pred_df = pd.DataFrame(predictions, columns=dk.label_list, index=features.index)
        
        # Calculate prediction confidence (using feature importance weighted prediction)
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            feature_importance = model.feature_importances_
            
            # Calculate weighted prediction confidence
            weighted_confidence = np.abs(predictions) * np.mean(feature_importance)
            pred_df['confidence'] = weighted_confidence
        else:
            # If feature importance is not available, use absolute prediction
            pred_df['confidence'] = np.abs(predictions)
        
        return pred_df

    def save_feature_importance(self, model: Any, dk: FreqaiDataKitchen) -> None:
        """
        Save feature importance to disk for analysis
        
        :param model: Trained model
        :param dk: FreqaiDataKitchen object
        """
        if hasattr(model, 'feature_importances_'):
            # Get feature names and importance
            feature_names = dk.data_dictionary['train_features'].columns
            importance = model.feature_importances_
            
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Save to disk
            dk.data_path / 'feature_importance.csv'
            feature_importance.to_csv(dk.data_path / 'feature_importance.csv', index=False)
            
            # Log top features
            top_features = feature_importance.head(10)
            logger.info(f"Top 10 features: {top_features.to_string()}")
        else:
            logger.warning("Model does not have feature_importances_ attribute")

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        """
        Fit the predictions to live data
        
        :param dk: FreqaiDataKitchen object
        :param pair: Trading pair
        """
        # Get the model
        model = self.model
        
        # Get the features
        features = dk.data_dictionary['prediction_features']
        
        # Make predictions
        predictions = model.predict(features)
        
        # Add predictions to the DataFrame
        pred_df = pd.DataFrame(predictions, columns=dk.label_list, index=features.index)
        
        # Calculate prediction confidence
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            feature_importance = model.feature_importances_
            
            # Calculate weighted prediction confidence
            weighted_confidence = np.abs(predictions) * np.mean(feature_importance)
            pred_df['confidence'] = weighted_confidence
        else:
            # If feature importance is not available, use absolute prediction
            pred_df['confidence'] = np.abs(predictions)
        
        # Add predictions to the data kitchen
        dk.data_dictionary['prediction_features'] = pred_df 