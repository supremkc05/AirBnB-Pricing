# 🏠 Airbnb Pricing Analysis Project

A comprehensive data analysis project focused on understanding Airbnb pricing patterns and building predictive models for property pricing optimization.

## 📋 Project Overview

This project analyzes Airbnb listing data to uncover insights about pricing strategies, location factors, and property characteristics that influence rental prices. The analysis includes data cleaning, exploratory data analysis (EDA), and potentially machine learning models for price prediction.

## 📁 Project Structure

```
Air-BnB-Pricing/
├── data/
│   └── airbnb.csv              # Raw Airbnb dataset
├── notebook/
│   ├── cleaning.ipynb          # Data cleaning and preprocessing
│   └── eda.ipynb              # Exploratory data analysis
├── dashboard/                  # (Planned) Interactive dashboard files
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```
## ⚙️ Feature Engineering Implemented

### 🔧 Data Cleaning & Preprocessing
- **Column Removal**: Dropped irrelevant columns (`id`, `license`, `neighbourhood_group`, `host_name`)
- **Missing Value Handling**:
  - Filled missing `price` values with median
  - Filled missing `reviews_per_month` with 0 (logical for properties with no reviews)
- **Date Conversion**: Converted `last_review` to datetime format

#### 1. **Temporal Features**
#### 2. **Price Categorization**
#### 3. **Economic Efficiency Metrics**
#### 4. **Premium Classification**
#### 5. **Price Normalization**

## 📊 Key Findings: Availability Impact Analysis

### 🎯 Primary Insights:

1. **Inverse Relationship**: Properties with **lower availability** tend to command **higher prices**
   - Scarcity
 drives premium pricing
   - High-demand properties are naturally less available

2. **Availability Categories**:
   - **Low Availability (0-90 days)**: Premium pricing tier
   - **Medium Availability (91-180 days)**: Balanced pricing
   - **High Availability (181-270 days)**: Competitive pricing
   - **Very High Availability (271-365 days)**: Value pricing

3. **Room Type Patterns**:
   - **Entire homes/apartments**: Lower average availability, higher prices
   - **Private rooms**: Moderate availability and pricing
   - **Shared rooms**: Higher availability, lower prices

### 💼 Business Implications:

- **Pricing Strategy**: Hosts can optimize revenue by strategic availability management
- **Market Positioning**: Low availability signals exclusivity and quality
- **Revenue Optimization**: Balance between occupancy rate and price premium
- **Geographic Hotspots**: Certain neighborhoods show distinct availability-price patterns

### 🔍 Statistical Evidence:
- Correlation coefficient shows the strength of availability-price relationship
- Price differences between high/low availability segments quantified
- Distribution patterns reveal market segmentation opportunities