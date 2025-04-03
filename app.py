import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ✅ Load Data Function (Handles Missing Files)
@st.cache_data
def load_data():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))  # Ensure correct path
        df_demand = pd.read_csv(os.path.join(base_path, "demand_forecasting.csv"))
        df_inventory = pd.read_csv(os.path.join(base_path, "inventory_monitoring.csv"))
        df_pricing = pd.read_csv(os.path.join(base_path, "pricing_optimization.csv"))
        return df_demand, df_inventory, df_pricing
    except FileNotFoundError:
        st.error("🚨 CSV files not found! Please upload `demand_forecasting.csv`, `inventory_monitoring.csv`, and `pricing_optimization.csv` to the project folder.")
        return None, None, None

# ✅ Load Data
df_demand, df_inventory, df_pricing = load_data()

# ✅ Streamlit UI
st.title("📊 AI-Powered Demand, Inventory & Pricing Optimization")

# 🚀 Demand Forecasting
st.subheader("📈 Demand Forecasting")
if df_demand is not None:
    st.write(df_demand.head())
    # Train simple Random Forest model
    if "Sales Quantity" in df_demand.columns:
        X = df_demand.drop(columns=["Sales Quantity"])
        y = df_demand["Sales Quantity"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X.select_dtypes(include=[np.number]), y)
        st.success("✅ Demand Forecasting Model Trained!")

# 📦 Inventory Monitoring
st.subheader("📦 Inventory Monitoring")
if df_inventory is not None:
    st.write(df_inventory.head())

# 💰 Pricing Optimization
st.subheader("💰 Pricing Optimization")
if df_pricing is not None:
    st.write(df_pricing.head())

st.sidebar.header("🔄 Upload New Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    st.write("📂 New Uploaded Data Preview:")
    st.write(new_df.head())
