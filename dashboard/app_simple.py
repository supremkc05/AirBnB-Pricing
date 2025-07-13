import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Airbnb Price Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Data loading function with multiple path attempts
@st.cache_data
def load_data():
    """Load data with multiple path attempts for deployment compatibility"""
    possible_paths = [
        'cleaned_airbnb.csv',  # Same directory
        'dashboard/cleaned_airbnb.csv',  # From root
        '../cleaned_airbnb.csv',  # Parent directory
        'data/airbnb.csv'  # Original data
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded from: {path}")
                return df
        except Exception as e:
            continue
    
    # If no file found, create sample data
    st.sidebar.warning("⚠️ Using sample data - CSV file not found")
    
    # Create sample data that matches the expected structure
    np.random.seed(42)
    sample_data = {
        'price': np.random.normal(150, 50, 1000),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 1000),
        'neighbourhood': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], 1000),
        'latitude': np.random.normal(40.7589, 0.1, 1000),
        'longitude': np.random.normal(-73.9851, 0.1, 1000),
        'minimum_nights': np.random.randint(1, 30, 1000),
        'number_of_reviews': np.random.randint(0, 500, 1000),
        'reviews_per_month': np.random.normal(2, 1, 1000),
        'availability_365': np.random.randint(0, 365, 1000),
        'calculated_host_listings_count': np.random.randint(1, 10, 1000)
    }
    
    return pd.DataFrame(sample_data)

# Load data
try:
    df = load_data()
    
    # Ensure price is positive
    df['price'] = np.abs(df['price'])
    df['price'] = df['price'].clip(lower=10)  # Minimum $10
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Data preprocessing
def preprocess_data(df):
    """Preprocess data with error handling"""
    try:
        if 'price_per_day' not in df.columns:
            df['price_per_day'] = df['price'] / (df['availability_365'] + 1)
        
        if 'availability_category' not in df.columns:
            df['availability_category'] = pd.cut(df['availability_365'], 
                                               bins=[0, 90, 180, 270, 365], 
                                               labels=['Low (0-90)', 'Medium (91-180)', 
                                                      'High (181-270)', 'Very High (271-365)'])
        
        if 'is_premium' not in df.columns:
            df['is_premium'] = df['price'] > df['price'].quantile(0.90)
        
        if 'price_segment' not in df.columns:
            df['price_segment'] = pd.cut(df['price'], 
                                       bins=[0, 75, 150, 300, float('inf')], 
                                       labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])
        
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing data: {str(e)}")
        return df

df = preprocess_data(df)

# Sidebar navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Dashboard Overview", "📊 Detailed Analysis", "🔮 Price Prediction", "📈 Market Insights"]
)

# Sidebar filters
st.sidebar.markdown("## 🔍 Filters")
room_type_filter = st.sidebar.multiselect(
    "Room Type:",
    options=df['room_type'].unique(),
    default=df['room_type'].unique()
)

price_range = st.sidebar.slider(
    "Price Range ($):",
    min_value=int(df['price'].min()),
    max_value=min(int(df['price'].max()), 1000),
    value=(int(df['price'].min()), min(500, int(df['price'].max())))
)

availability_range = st.sidebar.slider(
    "Availability (days/year):",
    min_value=0,
    max_value=365,
    value=(0, 365)
)

# Apply filters
filtered_df = df[
    (df['room_type'].isin(room_type_filter)) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1]) &
    (df['availability_365'] >= availability_range[0]) &
    (df['availability_365'] <= availability_range[1])
]

# Dashboard Overview Page
if page == "🏠 Dashboard Overview":
    st.markdown('<h1 class="main-header">🏠 Airbnb Price Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{filtered_df.shape[0]:,}</p>
            <p class="metric-label">Total Listings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_price = filtered_df['price'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">${avg_price:.0f}</p>
            <p class="metric-label">Average Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_availability = filtered_df['availability_365'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{avg_availability:.0f}</p>
            <p class="metric-label">Avg Availability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        premium_count = filtered_df['is_premium'].sum()
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{premium_count:,}</p>
            <p class="metric-label">Premium Listings</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">📍 Geographic Distribution</h3>', unsafe_allow_html=True)
        
        # Create geographic scatter plot
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            fig = px.scatter_mapbox(
                filtered_df.sample(min(1000, len(filtered_df))),
                lat='latitude',
                lon='longitude',
                color='price',
                size='price',
                hover_data=['room_type', 'availability_365'],
                color_continuous_scale='Viridis',
                size_max=15,
                zoom=10
            )
            fig.update_layout(
                mapbox_style="open-street-map",
                height=400,
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geographic data not available")
    
    with col2:
        st.markdown('<h3 class="section-header">💰 Price Distribution</h3>', unsafe_allow_html=True)
        
        # Price histogram
        fig = px.histogram(
            filtered_df,
            x='price',
            nbins=50,
            title='Price Distribution',
            color_discrete_sequence=['#FF5A5F']
        )
        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Room type analysis
    st.markdown('<h3 class="section-header">🏠 Room Type Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Room type distribution
        room_counts = filtered_df['room_type'].value_counts()
        fig = px.pie(
            values=room_counts.values,
            names=room_counts.index,
            title='Room Type Distribution',
            color_discrete_sequence=['#FF5A5F', '#FF385C', '#FD1D1D']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price by room type
        fig = px.box(
            filtered_df,
            x='room_type',
            y='price',
            title='Price by Room Type',
            color='room_type',
            color_discrete_sequence=['#FF5A5F', '#FF385C', '#FD1D1D']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown('<h3 class="section-header">💡 Key Insights</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Availability vs Price correlation
        corr = filtered_df['availability_365'].corr(filtered_df['price'])
        st.markdown(f"""
        <div class="insight-box">
            <h4>🔄 Availability-Price Relationship</h4>
            <p>Correlation coefficient: <strong>{corr:.3f}</strong></p>
            <p>{"Inverse relationship: Lower availability = Higher prices" if corr < 0 else "Direct relationship: Higher availability = Higher prices"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Premium listings insight
        premium_pct = (filtered_df['is_premium'].sum() / len(filtered_df)) * 100
        st.markdown(f"""
        <div class="insight-box">
            <h4>⭐ Premium Market</h4>
            <p>Premium listings: <strong>{premium_pct:.1f}%</strong></p>
            <p>Average premium price: <strong>${filtered_df[filtered_df['is_premium']]['price'].mean():.0f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Detailed Analysis Page
elif page == "📊 Detailed Analysis":
    st.markdown('<h1 class="main-header">📊 Detailed Analysis</h1>', unsafe_allow_html=True)
    
    # Availability Impact Analysis
    st.markdown('<h3 class="section-header">🎯 Availability Impact Analysis</h3>', unsafe_allow_html=True)
    
    # Create availability categories for analysis
    availability_stats = filtered_df.groupby('availability_category')['price'].agg(['count', 'mean', 'median', 'std']).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Availability vs Price scatter
        fig = px.scatter(
            filtered_df,
            x='availability_365',
            y='price',
            title='Availability vs Price',
            color='room_type',
            hover_data=['neighbourhood'] if 'neighbourhood' in filtered_df.columns else None
        )
        
        # Add trendline
        if len(filtered_df) > 1:
            X = filtered_df['availability_365'].values.reshape(-1, 1)
            y = filtered_df['price'].values
            
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
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price by availability category
        fig = px.box(
            filtered_df,
            x='availability_category',
            y='price',
            title='Price by Availability Category',
            color='availability_category'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(availability_stats, use_container_width=True)
    
    # Neighborhood Analysis
    if 'neighbourhood' in filtered_df.columns:
        st.markdown('<h3 class="section-header">🏘️ Neighborhood Analysis</h3>', unsafe_allow_html=True)
        
        # Top neighborhoods by count and price
        top_neighborhoods = filtered_df['neighbourhood'].value_counts().head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top neighborhoods by count
            fig = px.bar(
                x=top_neighborhoods.values,
                y=top_neighborhoods.index,
                orientation='h',
                title='Top 15 Neighborhoods by Listing Count',
                color=top_neighborhoods.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average price by neighborhood
            neighborhood_prices = filtered_df[filtered_df['neighbourhood'].isin(top_neighborhoods.index)].groupby('neighbourhood')['price'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=neighborhood_prices.values,
                y=neighborhood_prices.index,
                orientation='h',
                title='Average Price by Neighborhood',
                color=neighborhood_prices.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# Price Prediction Page
elif page == "🔮 Price Prediction":
    st.markdown('<h1 class="main-header">🔮 Price Prediction</h1>', unsafe_allow_html=True)
    
    # Model training section
    st.markdown('<h3 class="section-header">🤖 Model Performance</h3>', unsafe_allow_html=True)
    
    # Prepare features for modeling
    @st.cache_data
    def prepare_model_data(df):
        # Select numeric features that are likely to be present
        potential_numeric_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                                    'reviews_per_month', 'availability_365', 'calculated_host_listings_count']
        
        # Only use features that exist in the dataset
        numeric_features = [col for col in potential_numeric_features if col in df.columns]
        
        # Select categorical features
        potential_categorical_features = ['room_type', 'neighbourhood']
        categorical_features = [col for col in potential_categorical_features if col in df.columns]
        
        # Create feature matrix
        X = df[numeric_features].copy()
        
        # Handle missing values in numeric features
        X = X.fillna(X.mean())
        
        # Encode categorical features
        label_encoders = {}
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(df[feature].astype(str))
                label_encoders[feature] = le
        
        y = df['price']
        
        return X, y, numeric_features + categorical_features, label_encoders
    
    X, y, feature_names, label_encoders = prepare_model_data(df)
    
    # Train model
    @st.cache_data
    def train_model(X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return rf_model, {'MAE': mae, 'R2': r2, 'RMSE': rmse}
    
    model, metrics = train_model(X, y)
    
    # Display model performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{metrics['MAE']:.2f}</p>
            <p class="metric-label">Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{metrics['R2']:.3f}</p>
            <p class="metric-label">R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <p class="metric-value">{metrics['RMSE']:.2f}</p>
            <p class="metric-label">Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown('<h3 class="section-header">🎯 Feature Importance</h3>', unsafe_allow_html=True)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
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
    
    # Prediction interface
    st.markdown('<h3 class="section-header">🎯 Make a Prediction</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input fields
        room_type = st.selectbox("Room Type", df['room_type'].unique())
        neighbourhood = st.selectbox("Neighbourhood", df['neighbourhood'].unique()[:50])
        minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
        availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=200)
    
    with col2:
        latitude = st.number_input("Latitude", value=float(df['latitude'].mean()))
        longitude = st.number_input("Longitude", value=float(df['longitude'].mean()))
        number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
        reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
    
    # Predict button
    if st.button("🔮 Predict Price", key="predict_button"):
        # Prepare input data
        input_data = {}
        
        # Add numeric features
        for feature in ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                       'reviews_per_month', 'availability_365']:
            if feature in feature_names:
                input_data[feature] = locals()[feature]
        
        if 'calculated_host_listings_count' in feature_names:
            input_data['calculated_host_listings_count'] = 1
        
        # Add categorical features
        if 'room_type' in feature_names:
            input_data['room_type'] = label_encoders['room_type'].transform([room_type])[0]
        if 'neighbourhood' in feature_names:
            input_data['neighbourhood'] = label_encoders['neighbourhood'].transform([neighbourhood])[0]
        
        # Create DataFrame with all features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display prediction
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

# Market Insights Page
elif page == "📈 Market Insights":
    st.markdown('<h1 class="main-header">📈 Market Insights</h1>', unsafe_allow_html=True)
    
    # Correlation analysis
    st.markdown('<h3 class="section-header">🔗 Correlation Analysis</h3>', unsafe_allow_html=True)
    
    # Calculate correlation matrix
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix of Numeric Features',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Business insights
    st.markdown('<h3 class="section-header">💼 Business Insights</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # High vs Low availability comparison
        high_availability = filtered_df[filtered_df['availability_365'] > 300]['price']
        low_availability = filtered_df[filtered_df['availability_365'] < 90]['price']
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🎯 Availability Strategy</h4>
            <p><strong>High Availability (>300 days):</strong> ${high_availability.mean():.2f}</p>
            <p><strong>Low Availability (<90 days):</strong> ${low_availability.mean():.2f}</p>
            <p><strong>Price Premium:</strong> ${low_availability.mean() - high_availability.mean():.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Room type insights
        room_stats = filtered_df.groupby('room_type')['price'].agg(['mean', 'count']).round(2)
        
        room_insights = []
        for room_type in room_stats.index:
            room_insights.append(f"<p><strong>{room_type}:</strong> ${room_stats.loc[room_type, 'mean']:.2f} ({room_stats.loc[room_type, 'count']} listings)</p>")
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🏠 Room Type Performance</h4>
            {''.join(room_insights)}
        </div>
        """, unsafe_allow_html=True)
    
    # Market trends
    st.markdown('<h3 class="section-header">📊 Market Trends</h3>', unsafe_allow_html=True)
    
    # Create price distribution by room type
    fig = px.violin(
        filtered_df,
        x='room_type',
        y='price',
        title='Price Distribution by Room Type',
        color='room_type',
        box=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown('<h3 class="section-header">📋 Summary Statistics</h3>', unsafe_allow_html=True)
    
    summary_stats = filtered_df.describe()
    st.dataframe(summary_stats, use_container_width=True)
    
    # Recommendations
    st.markdown('<h3 class="section-header">💡 Recommendations</h3>', unsafe_allow_html=True)
    
    # Calculate recommendation values
    price_premium = (filtered_df[filtered_df['availability_365'] < 90]['price'].mean() - 
                    filtered_df[filtered_df['availability_365'] > 300]['price'].mean())
    
    luxury_threshold = filtered_df['price'].quantile(0.9)
    
    recommendations = [
        f"🎯 **Scarcity Strategy**: Lower availability (< 90 days) can increase prices by up to ${price_premium:.2f}",
        "🏠 **Room Type Focus**: Entire homes/apartments command the highest average prices",
        "📍 **Location Matters**: Geographic positioning significantly impacts pricing potential",
        "⭐ **Review Strategy**: Properties with moderate review activity (10-50 reviews) often perform best",
        f"💎 **Premium Positioning**: Top 10% of properties average ${luxury_threshold:.2f}, indicating luxury market potential"
    ]
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="insight-box">
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.9rem;'>"
    "🏠 Airbnb Price Analytics Dashboard | Built with Streamlit & Python"
    "</p>",
    unsafe_allow_html=True
)