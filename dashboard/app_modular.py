"""
Main modular Airbnb Price Analytics Dashboard.
This is the entry point for the modular version of the dashboard.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Configure the page
from src.config.settings import PAGE_CONFIG
st.set_page_config(
    page_title=PAGE_CONFIG['title'],
    page_icon=PAGE_CONFIG['icon'],
    layout=PAGE_CONFIG['layout'],
    initial_sidebar_state=PAGE_CONFIG['sidebar_state']
)

# Import components and pages
from src.components.header import render_header, render_footer
from src.components.sidebar import render_navigation, render_filters
from src.utils.data_preprocessing import load_data, apply_filters
from src.pages.dashboard import render_dashboard_page
from src.pages.analysis import render_analysis_page
from src.pages.prediction import render_prediction_page
from src.pages.market_insights import render_market_insights_page


def main():
    """Main application function."""
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    # Render header
    render_header()
    
    # Render navigation
    page = render_navigation()
    
    # Render filters
    room_type_filter, price_range, availability_range = render_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, room_type_filter, price_range, availability_range)
    
    # Render the selected page
    if page == "🏠 Dashboard Overview":
        render_dashboard_page(filtered_df)
    elif page == "📊 Detailed Analysis":
        render_analysis_page(filtered_df)
    elif page == "🔮 Price Prediction":
        render_prediction_page(df)  # Use full dataset for model training
    elif page == "📈 Market Insights":
        render_market_insights_page(filtered_df)
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
