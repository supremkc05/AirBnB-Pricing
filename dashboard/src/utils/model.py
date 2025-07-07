"""
Model utilities for the Airbnb Price Analytics Dashboard.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os


def load_saved_model():
    """Load the saved Random Forest model."""
    try:
        model_path = os.path.join('..', 'models', 'random_forest_model.pkl')
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def train_model(X, y):
    """Train a Random Forest model and return metrics."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rf_model, {'MAE': mae, 'R2': r2, 'RMSE': rmse}


def make_prediction(model, input_data):
    """Make a price prediction using the trained model."""
    try:
        prediction = model.predict(input_data)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


def get_feature_importance(model, feature_names):
    """Get feature importance from the trained model."""
    try:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return feature_importance
    except Exception as e:
        st.error(f"Error getting feature importance: {str(e)}")
        return None


def calculate_confidence_interval(prediction, confidence_level=0.15):
    """Calculate confidence interval for prediction."""
    confidence_interval = prediction * confidence_level
    return prediction - confidence_interval, prediction + confidence_interval
