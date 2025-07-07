"""
Configuration settings for the Airbnb Price Analytics Dashboard.
"""

import os

# Model configuration
MODEL_CONFIG = {
    'model_path': os.path.join('..', 'models', 'random_forest_model.pkl'),
    'model_type': 'random_forest',
    'target_column': 'price'
}

# Color scheme
COLOR_SCHEME = {
    'primary': '#FF5A5F',
    'secondary': '#FF385C',
    'tertiary': '#FD1D1D',
    'dark': '#8B0000',
    'gradient_start': '#FF5A5F',
    'gradient_end': '#FF385C'
}

# Chart configuration
CHART_CONFIG = {
    'color_sequence': ['#FF5A5F', '#FF385C', '#FD1D1D', '#8B0000'],
    'color_continuous_scale': 'Viridis',
    'default_height': 400,
    'map_style': 'open-street-map'
}

# Page configuration
PAGE_CONFIG = {
    'title': 'Airbnb Price Analytics',
    'icon': '🏠',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# Data configuration
DATA_CONFIG = {
    'data_file': 'cleaned_airbnb.csv',
    'max_price_filter': 1000,
    'max_sample_size': 1000,
    'top_neighborhoods': 15
}

# CSS styles
CSS_STYLES = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF5A5F;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #484848;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #FF5A5F;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #FF5A5F, #FF385C);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FF5A5F;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .insight-box h4 {
        color: #2c3e50 !important;
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .insight-box p {
        color: #2c3e50 !important;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    .insight-box strong {
        color: #FF5A5F !important;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF5A5F, #FF385C);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
</style>
"""
