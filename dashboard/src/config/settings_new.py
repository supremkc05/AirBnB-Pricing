"""
Configuration settings for Airbnb Price Analytics Dashboard
"""

# Color palette
COLORS = {
    'primary': '#FF5A5F',
    'secondary': '#FF385C', 
    'accent': '#FD1D1D',
    'dark': '#484848',
    'light': '#f5f7fa',
    'text': '#2c3e50'
}

# Chart configuration
CHART_CONFIG = {
    'color_sequence': ['#FF5A5F', '#FF385C', '#FD1D1D', '#8B0000'],
    'color_scale': 'Viridis',
    'height': 400,
    'margin': {"r": 0, "t": 0, "l": 0, "b": 0}
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "Airbnb Price Analytics",
    "page_icon": "🏠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Pages configuration
PAGES = {
    "🏠 Dashboard Overview": "dashboard",
    "📊 Detailed Analysis": "analysis", 
    "🔮 Price Prediction": "prediction",
    "📈 Market Insights": "insights"
}

# Model configuration
MODEL_CONFIG = {
    'model_path': '../../models/random_forest_model.pkl',
    'encoders_path': '../../models/label_encoders.pkl',
    'features_path': '../../models/feature_names.pkl'
}
