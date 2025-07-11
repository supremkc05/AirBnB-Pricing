"""
Data preprocessing utilities for the Airbnb Price Analytics Dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import os


def load_data(file_path=None):
    """Load and preprocess the Airbnb dataset."""
    if file_path is None:
        # Try multiple possible paths for the data file
        possible_paths = [
            'cleaned_airbnb.csv',  # Same directory as app
            '../data/cleaned_airbnb.csv',  # Parent data folder
            'data/cleaned_airbnb.csv',  # Data folder from root
            os.path.join(os.path.dirname(__file__), '..', '..', 'cleaned_airbnb.csv'),  # Dashboard level
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'cleaned_airbnb.csv')  # Project level
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        else:
            st.error("Data file 'cleaned_airbnb.csv' not found in any expected location.")
            st.info("Please ensure 'cleaned_airbnb.csv' is in the dashboard directory or data folder.")
            return None
    
    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Data loaded successfully from: {file_path}")
        return preprocess_data(df)
    except FileNotFoundError:
        st.error(f"Data file '{file_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess_data(df):
    """Apply preprocessing steps to the dataframe."""
    df = df.copy()
    
    # Create derived features
    if 'price_per_day' not in df.columns:
        df['price_per_day'] = df['price'] / (df['availability_365'] + 1)
    
    if 'availability_category' not in df.columns:
        df['availability_category'] = pd.cut(
            df['availability_365'], 
            bins=[0, 90, 180, 270, 365], 
            labels=['Low (0-90)', 'Medium (91-180)', 'High (181-270)', 'Very High (271-365)']
        )
    
    if 'is_premium' not in df.columns:
        df['is_premium'] = df['price'] > df['price'].quantile(0.90)
    
    if 'price_segment' not in df.columns:
        df['price_segment'] = pd.cut(
            df['price'], 
            bins=[0, 75, 150, 300, float('inf')], 
            labels=['Budget', 'Mid-range', 'Premium', 'Luxury']
        )
    
    return df


def apply_filters(df, room_type_filter, price_range, availability_range):
    """Apply user-selected filters to the dataframe."""
    return df[
        (df['room_type'].isin(room_type_filter)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1]) &
        (df['availability_365'] >= availability_range[0]) &
        (df['availability_365'] <= availability_range[1])
    ]


def prepare_model_features(df):
    """Prepare features for machine learning models."""
    # Select numeric features that are likely to be present
    potential_numeric_features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
        'reviews_per_month', 'availability_365', 'calculated_host_listings_count'
    ]
    
    # Only use features that exist in the dataset
    numeric_features = [col for col in potential_numeric_features if col in df.columns]
    
    # Select categorical features
    potential_categorical_features = ['room_type', 'neighbourhood']
    categorical_features = [col for col in potential_categorical_features if col in df.columns]
    
    # Create feature matrix
    X = df[numeric_features].copy()
    
    # Handle missing values in numeric features
    X = X.fillna(X.mean())
    
    # Encode categorical features
    label_encoders = {}
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            X[feature] = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le
    
    y = df['price']
    feature_names = numeric_features + categorical_features
    
    return X, y, feature_names, label_encoders


def prepare_prediction_input(input_data, feature_names, label_encoders):
    """Prepare input data for prediction."""
    # Create DataFrame with all features
    input_df = pd.DataFrame([input_data])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value for missing features
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    return input_df


def get_similar_properties(df, filters, limit=5):
    """Get similar properties based on filters."""
    similar_properties = df.copy()
    
    for key, value in filters.items():
        if key in similar_properties.columns and value is not None:
            similar_properties = similar_properties[similar_properties[key] == value]
    
    return similar_properties.head(limit)
