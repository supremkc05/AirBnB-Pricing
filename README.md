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

## cleaned data saved in data folder
df.to_csv('../data/cleaned_airbnb.csv')

## 📊 Exploratory Data Analysis (EDA)
