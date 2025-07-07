"""
Metrics component for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st


def render_metric_card(value, label, format_type="number"):
    """Render a single metric card."""
    if format_type == "currency":
        formatted_value = f"${value:.0f}"
    elif format_type == "percentage":
        formatted_value = f"{value:.1f}%"
    else:
        formatted_value = f"{value:,.0f}"
    
    st.markdown(f"""
    <div class="metric-container">
        <p class="metric-value">{formatted_value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def render_key_metrics(filtered_df):
    """Render key metrics for the dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(filtered_df.shape[0], "Total Listings")
    
    with col2:
        avg_price = filtered_df['price'].mean()
        render_metric_card(avg_price, "Average Price", "currency")
    
    with col3:
        avg_availability = filtered_df['availability_365'].mean()
        render_metric_card(avg_availability, "Avg Availability")
    
    with col4:
        premium_count = filtered_df['is_premium'].sum()
        render_metric_card(premium_count, "Premium Listings")


def render_model_metrics(metrics):
    """Render model performance metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card(metrics['MAE'], "Mean Absolute Error", "currency")
    
    with col2:
        render_metric_card(metrics['R2'], "R² Score", "percentage")
    
    with col3:
        render_metric_card(metrics['RMSE'], "Root Mean Squared Error", "currency")


def render_prediction_result(prediction, confidence_interval):
    """Render prediction result in a styled box."""
    st.markdown(f"""
    <div class="prediction-box">
        <h2>🎯 Predicted Price</h2>
        <h1>${prediction:.2f}</h1>
        <p>Based on the provided property characteristics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show confidence interval
    low, high = confidence_interval
    st.success(f"💡 **Confidence Range:** ${low:.2f} - ${high:.2f}")


def render_insight_box(title, content):
    """Render an insight box with formatted content."""
    st.markdown(f"""
    <div class="insight-box">
        <h4>{title}</h4>
        {content}
    </div>
    """, unsafe_allow_html=True)
