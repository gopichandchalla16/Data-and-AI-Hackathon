import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def load_data():
    df_demand = pd.read_csv("demand_forecasting.csv")
    df_inventory = pd.read_csv("inventory_monitoring.csv")
    df_pricing = pd.read_csv("pricing_optimization.csv")
    return df_demand, df_inventory, df_pricing

def demand_forecasting(df):
    features = ['Price', 'Promotions', 'Seasonality Factors', 'External Factors']
    df = pd.get_dummies(df, columns=['Promotions', 'Seasonality Factors', 'External Factors'], drop_first=True)
    X = df[features]
    y = df['Sales Quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    return model, error

def inventory_monitoring(df):
    df['Stock Risk'] = np.where(df['Stock Levels'] < df['Reorder Point'], 'Reorder', 'Sufficient')
    return df[['Product ID', 'Store ID', 'Stock Levels', 'Reorder Point', 'Stock Risk']]

def pricing_optimization(df):
    df["Optimized Price"] = df["Price"] * 0.9
    return df[['Product ID', 'Store ID', 'Price', 'Optimized Price']]

st.title("Multi-Agent Supply Chain Optimization System")

df_demand, df_inventory, df_pricing = load_data()

option = st.sidebar.selectbox("Choose Model to Run", ["Demand Forecasting", "Inventory Monitoring", "Pricing Optimization"])

if option == "Demand Forecasting":
    st.subheader("Demand Forecasting Model")
    model, error = demand_forecasting(df_demand)
    st.write(f"Model trained! MAE: {error:.2f}")
    
elif option == "Inventory Monitoring":
    st.subheader("Inventory Monitoring Insights")
    inventory_results = inventory_monitoring(df_inventory)
    st.dataframe(inventory_results)

elif option == "Pricing Optimization":
    st.subheader("Pricing Optimization Results")
    pricing_results = pricing_optimization(df_pricing)
    st.dataframe(pricing_results)

st.sidebar.info("Upload updated datasets to retrain models.")
