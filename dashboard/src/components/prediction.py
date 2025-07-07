"""
Prediction component using saved model
"""
import streamlit as st
import pandas as pd
from ..utils.model import load_saved_model, make_prediction, get_feature_importance

def render_prediction_form():
    """Render the prediction input form"""
    st.markdown('<h3 class="section-header">🎯 Make a Prediction</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Room type input
        room_type = st.selectbox(
            "Room Type", 
            ["Private room", "Entire home/apt", "Shared room"]
        )
        
        # Neighbourhood input
        neighbourhood = st.text_input("Neighbourhood", value="Manhattan")
        
        # Minimum nights input
        minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
        
        # Availability input
        availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=200)
    
    with col2:
        # Location inputs
        latitude = st.number_input("Latitude", value=40.7589)
        longitude = st.number_input("Longitude", value=-73.9851)
        
        # Reviews input
        number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
        
        # Reviews per month input
        reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
    
    # Predict button
    if st.button("🔮 Predict Price", key="predict_button"):
        input_data = {
            'room_type': room_type,
            'neighbourhood': neighbourhood,
            'minimum_nights': minimum_nights,
            'availability_365': availability_365,
            'latitude': latitude,
            'longitude': longitude,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month
        }
        
        prediction = make_prediction(input_data)
        
        if prediction is not None:
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Predicted Price</h2>
                <h1>${prediction:.2f}</h1>
                <p>Based on the provided property characteristics</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence interval
            confidence_interval = prediction * 0.15
            st.success(f"💡 **Confidence Range:** ${prediction - confidence_interval:.2f} - ${prediction + confidence_interval:.2f}")
        else:
            st.error("❌ Unable to make prediction. Please check the model configuration.")

def render_model_info():
    """Render model information"""
    st.markdown('<h3 class="section-header">🤖 Model Information</h3>', unsafe_allow_html=True)
    
    model, _, _ = load_saved_model()
    
    if model is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{type(model).__name__}</p>
                <p class="metric-label">Model Type</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            n_estimators = getattr(model, 'n_estimators', 'N/A')
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{n_estimators}</p>
                <p class="metric-label">Estimators</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            feature_count = getattr(model, 'n_features_in_', 'N/A')
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value">{feature_count}</p>
                <p class="metric-label">Features</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ Model not loaded. Please check the model file.")

def render_feature_importance():
    """Render feature importance chart"""
    st.markdown('<h3 class="section-header">🎯 Feature Importance</h3>', unsafe_allow_html=True)
    
    feature_importance = get_feature_importance()
    
    if feature_importance is not None:
        import plotly.express as px
        
        fig = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Feature importance not available for this model type.")
