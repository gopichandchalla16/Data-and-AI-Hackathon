import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# **📌 Multi-Agent AI System Functions**
class DemandAgent:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        # Handle missing values
        self.data.fillna("Unknown", inplace=True)
        
        # Convert categorical columns to string and encode
        label_encoders = {}
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype(str)  # Convert to string
            label_encoders[col] = LabelEncoder()
            self.data[col] = label_encoders[col].fit_transform(self.data[col])
        
        return self.data

    def predict_demand(self):
        self.data = self.preprocess_data()
        
        try:
            required_cols = {'Price', 'Promotions', 'Seasonality Factors', 'Sales Quantity'}
            if not required_cols.issubset(self.data.columns):
                missing_cols = required_cols - set(self.data.columns)
                st.error(f"❌ Missing required columns: {missing_cols}")
                return None, None
            
            X = self.data[['Price', 'Promotions', 'Seasonality Factors']]
            y = self.data['Sales Quantity']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            self.data['Predicted Demand'] = model.predict(X)

            return self.data, mae

        except Exception as e:
            st.error(f"❌ Error in Demand Forecasting: {e}")
            return None, None

# **🚀 Streamlit UI**
st.title("🛒 AI-Powered Retail Inventory Optimization (Multi-Agent Framework)")
st.markdown("### **Upload Your Retail Data** (CSV Format)")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.write(df.head())

    # **Demand Forecasting**
    st.subheader("📈 Demand Forecasting")
    demand_agent = DemandAgent(df)
    df, mae = demand_agent.predict_demand()
    
    if df is not None:
        st.write(f"**Model MAE:** {mae:.2f}")
        fig_demand = px.line(df, x=df.index, y=["Sales Quantity", "Predicted Demand"], title="Actual vs. Predicted Demand")
        st.plotly_chart(fig_demand)

    st.success("✅ App Successfully Deployed!")
else:
    st.warning("⚠️ Please upload a CSV file.")
