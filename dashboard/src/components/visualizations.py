"""
Visualizations component for the Airbnb Price Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from ..config.settings import CHART_CONFIG, DATA_CONFIG


def create_geographic_map(df):
    """Create a geographic scatter plot map."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.info("Geographic data not available")
        return None
    
    # Sample data for performance
    sample_size = min(DATA_CONFIG['max_sample_size'], len(df))
    sample_df = df.sample(sample_size)
    
    fig = px.scatter_mapbox(
        sample_df,
        lat='latitude',
        lon='longitude',
        color='price',
        size='price',
        hover_data=['room_type', 'availability_365'],
        color_continuous_scale=CHART_CONFIG['color_continuous_scale'],
        size_max=15,
        zoom=10,
        title='Geographic Distribution of Listings'
    )
    
    fig.update_layout(
        mapbox_style=CHART_CONFIG['map_style'],
        height=CHART_CONFIG['default_height'],
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig


def create_price_histogram(df):
    """Create a price distribution histogram."""
    fig = px.histogram(
        df,
        x='price',
        nbins=50,
        title='Price Distribution',
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    fig.update_layout(
        xaxis_title="Price ($)",
        yaxis_title="Count",
        height=CHART_CONFIG['default_height']
    )
    
    return fig


def create_room_type_pie(df):
    """Create a room type distribution pie chart."""
    room_counts = df['room_type'].value_counts()
    
    fig = px.pie(
        values=room_counts.values,
        names=room_counts.index,
        title='Room Type Distribution',
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    return fig


def create_price_by_room_type_box(df):
    """Create a box plot of price by room type."""
    fig = px.box(
        df,
        x='room_type',
        y='price',
        title='Price by Room Type',
        color='room_type',
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    fig.update_layout(height=CHART_CONFIG['default_height'])
    
    return fig


def create_availability_price_scatter(df):
    """Create availability vs price scatter plot with trendline."""
    fig = px.scatter(
        df,
        x='availability_365',
        y='price',
        title='Availability vs Price',
        color='room_type',
        hover_data=['neighbourhood'] if 'neighbourhood' in df.columns else None,
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    # Add trendline
    if len(df) > 1:
        X = df['availability_365'].values.reshape(-1, 1)
        y = df['price'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) > 1:
            lr = LinearRegression()
            lr.fit(X_clean, y_clean)
            
            # Create trendline
            x_range = np.linspace(X_clean.min(), X_clean.max(), 100)
            y_pred = lr.predict(x_range.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_pred,
                mode='lines',
                name='Trendline',
                line=dict(color='red', width=2)
            ))
    
    fig.update_layout(height=CHART_CONFIG['default_height'])
    
    return fig


def create_price_by_availability_category_box(df):
    """Create a box plot of price by availability category."""
    fig = px.box(
        df,
        x='availability_category',
        y='price',
        title='Price by Availability Category',
        color='availability_category',
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    fig.update_layout(height=CHART_CONFIG['default_height'])
    
    return fig


def create_neighborhood_analysis(df):
    """Create neighborhood analysis charts."""
    if 'neighbourhood' not in df.columns:
        return None, None
    
    # Top neighborhoods by count
    top_neighborhoods = df['neighbourhood'].value_counts().head(DATA_CONFIG['top_neighborhoods'])
    
    fig1 = px.bar(
        x=top_neighborhoods.values,
        y=top_neighborhoods.index,
        orientation='h',
        title=f'Top {DATA_CONFIG["top_neighborhoods"]} Neighborhoods by Listing Count',
        color=top_neighborhoods.values,
        color_continuous_scale=CHART_CONFIG['color_continuous_scale']
    )
    fig1.update_layout(height=500)
    
    # Average price by neighborhood
    neighborhood_prices = df[df['neighbourhood'].isin(top_neighborhoods.index)].groupby('neighbourhood')['price'].mean().sort_values(ascending=False)
    
    fig2 = px.bar(
        x=neighborhood_prices.values,
        y=neighborhood_prices.index,
        orientation='h',
        title='Average Price by Neighborhood',
        color=neighborhood_prices.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig2.update_layout(height=500)
    
    return fig1, fig2


def create_price_segment_pie(df):
    """Create price segment distribution pie chart."""
    segment_counts = df['price_segment'].value_counts()
    
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title='Market Share by Price Segment',
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    return fig


def create_segment_characteristics_bar(df):
    """Create segment characteristics bar chart."""
    segment_chars = df.groupby('price_segment')['availability_365'].mean()
    
    fig = px.bar(
        x=segment_chars.index,
        y=segment_chars.values,
        title='Average Availability by Price Segment',
        color=segment_chars.index,
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    return fig


def create_reviews_histogram(df):
    """Create number of reviews histogram."""
    fig = px.histogram(
        df,
        x='number_of_reviews',
        nbins=50,
        title='Distribution of Number of Reviews',
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    fig.update_layout(xaxis_range=[0, 200])
    
    return fig


def create_price_vs_reviews_scatter(df):
    """Create price vs reviews scatter plot."""
    fig = px.scatter(
        df,
        x='number_of_reviews',
        y='price',
        title='Price vs Number of Reviews',
        color='room_type',
        opacity=0.6,
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    fig.update_layout(xaxis_range=[0, 200], yaxis_range=[0, 500])
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap of numeric features."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix of Numeric Features',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    return fig


def create_price_violin_plot(df):
    """Create price distribution violin plot by room type."""
    fig = px.violin(
        df,
        x='room_type',
        y='price',
        title='Price Distribution by Room Type',
        color='room_type',
        box=True,
        color_discrete_sequence=CHART_CONFIG['color_sequence']
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_feature_importance_chart(feature_importance):
    """Create feature importance bar chart."""
    fig = px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='importance',
        color_continuous_scale=CHART_CONFIG['color_continuous_scale']
    )
    
    return fig
