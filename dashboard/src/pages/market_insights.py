"""
Market insights page for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
from ..components.header import render_page_header, render_section_header
from ..components.metrics import render_insight_box
from ..components.visualizations import create_correlation_heatmap, create_price_violin_plot


def render_market_insights_page(filtered_df):
    """Render the market insights page."""
    # Page header
    render_page_header("Market Insights", "📈")
    
    # Correlation analysis
    render_section_header("🔗 Correlation Analysis")
    
    fig = create_correlation_heatmap(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Business insights
    render_section_header("💼 Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # High vs Low availability comparison
        high_availability = filtered_df[filtered_df['availability_365'] > 300]['price']
        low_availability = filtered_df[filtered_df['availability_365'] < 90]['price']
        
        insight_content = f"""
        <p><strong>High Availability (>300 days):</strong> ${high_availability.mean():.2f}</p>
        <p><strong>Low Availability (<90 days):</strong> ${low_availability.mean():.2f}</p>
        <p><strong>Price Premium:</strong> ${low_availability.mean() - high_availability.mean():.2f}</p>
        """
        render_insight_box("🎯 Availability Strategy", insight_content)
    
    with col2:
        # Room type insights
        room_stats = filtered_df.groupby('room_type')['price'].agg(['mean', 'count']).round(2)
        
        room_insights = []
        for room_type in room_stats.index:
            room_insights.append(f"<p><strong>{room_type}:</strong> ${room_stats.loc[room_type, 'mean']:.2f} ({room_stats.loc[room_type, 'count']} listings)</p>")
        
        insight_content = ''.join(room_insights)
        render_insight_box("🏠 Room Type Performance", insight_content)
    
    # Market trends
    render_section_header("📊 Market Trends")
    
    fig = create_price_violin_plot(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    render_section_header("📋 Summary Statistics")
    
    summary_stats = filtered_df.describe()
    st.dataframe(summary_stats, use_container_width=True)
    
    # Recommendations
    render_section_header("💡 Recommendations")
    
    # Calculate recommendation values
    price_premium = (filtered_df[filtered_df['availability_365'] < 90]['price'].mean() - 
                    filtered_df[filtered_df['availability_365'] > 300]['price'].mean())
    
    luxury_threshold = filtered_df['price'].quantile(0.9)
    
    recommendations = [
        f"🎯 **Scarcity Strategy**: Lower availability (< 90 days) can increase prices by up to ${price_premium:.2f}",
        "🏠 **Room Type Focus**: Entire homes/apartments command the highest average prices",
        "📍 **Location Matters**: Geographic positioning significantly impacts pricing potential",
        "⭐ **Review Strategy**: Properties with moderate review activity (10-50 reviews) often perform best",
        f"🎨 **Premium Positioning**: Top 10% of properties average ${luxury_threshold:.2f}, indicating luxury market potential"
    ]
    
    for rec in recommendations:
        render_insight_box("", f"<p>{rec}</p>")
