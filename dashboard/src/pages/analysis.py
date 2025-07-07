"""
Detailed analysis page for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
from ..components.header import render_page_header, render_section_header
from ..components.visualizations import (
    create_availability_price_scatter,
    create_price_by_availability_category_box,
    create_neighborhood_analysis,
    create_price_segment_pie,
    create_segment_characteristics_bar,
    create_reviews_histogram,
    create_price_vs_reviews_scatter
)


def render_analysis_page(filtered_df):
    """Render the detailed analysis page."""
    # Page header
    render_page_header("Detailed Analysis", "📊")
    
    # Availability Impact Analysis
    render_section_header("🎯 Availability Impact Analysis")
    
    # Create availability categories for analysis
    availability_stats = filtered_df.groupby('availability_category')['price'].agg(['count', 'mean', 'median', 'std']).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_availability_price_scatter(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_price_by_availability_category_box(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(availability_stats, use_container_width=True)
    
    # Neighborhood Analysis
    render_section_header("🏘️ Neighborhood Analysis")
    
    fig1, fig2 = create_neighborhood_analysis(filtered_df)
    
    if fig1 and fig2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Neighborhood data not available for analysis")
    
    # Market Segmentation
    render_section_header("🎯 Market Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_price_segment_pie(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_segment_characteristics_bar(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Reviews Analysis
    render_section_header("⭐ Reviews Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_reviews_histogram(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_price_vs_reviews_scatter(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
