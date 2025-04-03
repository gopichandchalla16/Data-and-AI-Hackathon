import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# **📌 Multi-Agent AI System Functions**
class DemandAgent:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        # Handle missing values
        self.data.fillna(0, inplace=True)
        
        # Convert categorical columns to numeric using Label Encoding
        label_encoders = {}
        for col in self.data.select_dtypes(include=['object']).columns:
            label_encoders[col] = LabelEncoder()
            self.data[col] = label_encoders[col].fit_transform(self.data[col])
        
        return self.data

    def predict_demand(self):
        self.data = self.preprocess_data()
        
        try:
            X = self.data[['Price', 'Promotions', 'Seasonality Factors']]
            y = self.data['Sales Quantity']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            self.data['Predicted Demand'] = model.predict(X)

            return self.data, mae

        except ValueError as e:
            st.error(f"❌ Data type error: {e}. Ensure all columns have correct numeric values.")
            return None, None

class InventoryAgent:
    def __init__(self, data):
        self.data = data

    def monitor_inventory(self):
        self.data['Stock Risk'] = np.where(self.data['Stock Levels'] < self.data['Reorder Point'], 'Reorder', 'Sufficient')
        return self.data

class PricingAgent:
    def __init__(self, data):
        self.data = data

    def optimize_pricing(self):
        self.data['Optimized Price'] = self.data['Price'] * (1 - (self.data['Discounts'] / 100))
        return self.data

# **🚀 Streamlit UI**
st.title("🛒 AI-Powered Retail Inventory Optimization (Multi-Agent Framework)")
st.markdown("### **Upload Your Retail Data** (CSV Format)")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.write(df.head())

    # **Demand Forecasting**
    if {'Price', 'Promotions', 'Seasonality Factors', 'Sales Quantity'}.issubset(df.columns):
        st.subheader("📈 Demand Forecasting")
        demand_agent = DemandAgent(df)
        df, mae = demand_agent.predict_demand()
        
        if df is not None:
            st.write(f"**Model MAE:** {mae:.2f}")
            fig_demand = px.line(df, x=df.index, y=["Sales Quantity", "Predicted Demand"], title="Actual vs. Predicted Demand")
            st.plotly_chart(fig_demand)

    # **Inventory Monitoring**
    if {'Stock Levels', 'Reorder Point'}.issubset(df.columns):
        st.subheader("📦 Inventory Monitoring")
        inventory_agent = InventoryAgent(df)
        df = inventory_agent.monitor_inventory()
        st.write(df[['Product ID', 'Stock Levels', 'Reorder Point', 'Stock Risk']].head())

    # **Pricing Optimization**
    if {'Price', 'Discounts'}.issubset(df.columns):
        st.subheader("💰 Pricing Optimization")
        pricing_agent = PricingAgent(df)
        df = pricing_agent.optimize_pricing()
        st.write(df[['Product ID', 'Price', 'Discounts', 'Optimized Price']].head())

        fig_pricing = px.scatter(df, x="Price", y="Optimized Price", color="Price", title="Pricing Optimization Impact")
        st.plotly_chart(fig_pricing)

    st.success("✅ Multi-Agent AI System Processed the Data Successfully!")

else:
    st.warning("⚠️ Please upload a CSV file to analyze.")
