"""
Sidebar component for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
from ..config.settings import DATA_CONFIG


def render_navigation():
    """Render the navigation sidebar."""
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Dashboard Overview", "📊 Detailed Analysis", "🔮 Price Prediction", "📈 Market Insights"]
    )
    return page


def render_filters(df):
    """Render filter controls in the sidebar."""
    st.sidebar.markdown("## 🔍 Filters")
    
    # Room type filter
    room_type_filter = st.sidebar.multiselect(
        "Room Type:",
        options=df['room_type'].unique(),
        default=df['room_type'].unique()
    )
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($):",
        min_value=int(df['price'].min()),
        max_value=min(int(df['price'].max()), DATA_CONFIG['max_price_filter']),
        value=(int(df['price'].min()), min(500, int(df['price'].max())))
    )
    
    # Availability range filter
    availability_range = st.sidebar.slider(
        "Availability (days/year):",
        min_value=0,
        max_value=365,
        value=(0, 365)
    )
    
    return room_type_filter, price_range, availability_range


def render_info_box(title, content):
    """Render an information box in the sidebar."""
    st.sidebar.markdown(f"## {title}")
    st.sidebar.info(content)
