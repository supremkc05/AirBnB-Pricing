"""
Price prediction page for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
from ..components.header import render_page_header, render_section_header
from ..components.metrics import render_model_metrics, render_prediction_result
from ..components.visualizations import create_feature_importance_chart
from ..utils.data_preprocessing import prepare_model_features, prepare_prediction_input
from ..utils.model import train_model, make_prediction, get_feature_importance, calculate_confidence_interval


def render_prediction_page(df):
    """Render the price prediction page."""
    # Page header
    render_page_header("Price Prediction", "🔮")
    
    # Model training section
    render_section_header("🤖 Model Training")
    
    # Prepare features for modeling
    X, y, feature_names, label_encoders = prepare_model_features(df)
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Display model performance
    render_model_metrics(metrics)
    
    # Feature importance
    render_section_header("🎯 Feature Importance")
    
    feature_importance = get_feature_importance(model, feature_names)
    if feature_importance is not None:
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    render_section_header("🎯 Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input fields
        if 'room_type' in df.columns:
            room_type = st.selectbox("Room Type", df['room_type'].unique())
        else:
            room_type = None
            
        if 'neighbourhood' in df.columns:
            neighbourhood = st.selectbox("Neighbourhood", df['neighbourhood'].unique()[:50])  # Limit for performance
        else:
            neighbourhood = None
            
        minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
        availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=200)
    
    with col2:
        if 'latitude' in df.columns:
            latitude = st.number_input("Latitude", value=float(df['latitude'].mean()))
        else:
            latitude = None
            
        if 'longitude' in df.columns:
            longitude = st.number_input("Longitude", value=float(df['longitude'].mean()))
        else:
            longitude = None
            
        number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
        
        if 'reviews_per_month' in df.columns:
            reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
        else:
            reviews_per_month = None
    
    # Predict button
    if st.button("🔮 Predict Price", key="predict_button"):
        # Prepare input data
        input_data = {}
        
        # Add numeric features
        if 'latitude' in feature_names and latitude is not None:
            input_data['latitude'] = latitude
        if 'longitude' in feature_names and longitude is not None:
            input_data['longitude'] = longitude
        if 'minimum_nights' in feature_names:
            input_data['minimum_nights'] = minimum_nights
        if 'number_of_reviews' in feature_names:
            input_data['number_of_reviews'] = number_of_reviews
        if 'reviews_per_month' in feature_names and reviews_per_month is not None:
            input_data['reviews_per_month'] = reviews_per_month
        if 'availability_365' in feature_names:
            input_data['availability_365'] = availability_365
        if 'calculated_host_listings_count' in feature_names:
            input_data['calculated_host_listings_count'] = 1  # Default value
        
        # Add categorical features
        if 'room_type' in feature_names and room_type is not None:
            input_data['room_type'] = label_encoders['room_type'].transform([room_type])[0]
        if 'neighbourhood' in feature_names and neighbourhood is not None:
            input_data['neighbourhood'] = label_encoders['neighbourhood'].transform([neighbourhood])[0]
        
        # Prepare input DataFrame
        input_df = prepare_prediction_input(input_data, feature_names, label_encoders)
        
        # Make prediction
        prediction = make_prediction(model, input_df)
        
        if prediction is not None:
            # Calculate confidence interval
            confidence_interval = calculate_confidence_interval(prediction)
            
            # Display prediction
            render_prediction_result(prediction, confidence_interval)
            
            # Show similar properties
            st.markdown('<h4>🔍 Similar Properties</h4>', unsafe_allow_html=True)
            
            similar_properties = df.copy()
            if room_type:
                similar_properties = similar_properties[similar_properties['room_type'] == room_type]
            if neighbourhood:
                similar_properties = similar_properties[similar_properties['neighbourhood'] == neighbourhood]
            
            similar_properties = similar_properties.head(5)
            
            if not similar_properties.empty:
                display_cols = ['price', 'availability_365', 'number_of_reviews']
                if 'room_type' in similar_properties.columns:
                    display_cols.insert(0, 'room_type')
                if 'neighbourhood' in similar_properties.columns:
                    display_cols.insert(1, 'neighbourhood')
                
                st.dataframe(
                    similar_properties[display_cols],
                    use_container_width=True
                )
