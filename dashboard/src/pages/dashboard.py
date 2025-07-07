"""
Dashboard overview page for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
from ..components.header import render_page_header, render_section_header
from ..components.metrics import render_key_metrics, render_insight_box
from ..components.visualizations import (
    create_geographic_map, 
    create_price_histogram, 
    create_room_type_pie, 
    create_price_by_room_type_box
)


def render_dashboard_page(filtered_df):
    """Render the main dashboard overview page."""
    # Page header
    render_page_header("Dashboard Overview", "🏠")
    
    # Key metrics
    render_key_metrics(filtered_df)
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        render_section_header("📍 Geographic Distribution")
        fig = create_geographic_map(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        render_section_header("💰 Price Distribution")
        fig = create_price_histogram(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Room type analysis
    render_section_header("🏠 Room Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_room_type_pie(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_price_by_room_type_box(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    render_section_header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Availability vs Price correlation
        corr = filtered_df['availability_365'].corr(filtered_df['price'])
        insight_content = f"""
        <p>Correlation coefficient: <strong>{corr:.3f}</strong></p>
        <p>{"Inverse relationship: Lower availability = Higher prices" if corr < 0 else "Direct relationship: Higher availability = Higher prices"}</p>
        """
        render_insight_box("🔄 Availability-Price Relationship", insight_content)
    
    with col2:
        # Premium listings insight
        premium_pct = (filtered_df['is_premium'].sum() / len(filtered_df)) * 100
        premium_avg_price = filtered_df[filtered_df['is_premium']]['price'].mean()
        insight_content = f"""
        <p>Premium listings: <strong>{premium_pct:.1f}%</strong></p>
        <p>Average premium price: <strong>${premium_avg_price:.0f}</strong></p>
        """
        render_insight_box("⭐ Premium Market", insight_content)
