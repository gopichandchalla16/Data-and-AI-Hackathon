import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI  # For AI Chat Agent (Requires API Key)

# ✅ Load Data Function (Handles Missing Files)
@st.cache_data
def load_data():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))  
        df_demand = pd.read_csv(os.path.join(base_path, "demand_forecasting.csv"))
        df_inventory = pd.read_csv(os.path.join(base_path, "inventory_monitoring.csv"))
        df_pricing = pd.read_csv(os.path.join(base_path, "pricing_optimization.csv"))
        return df_demand, df_inventory, df_pricing
    except FileNotFoundError:
        st.error("🚨 CSV files not found! Please upload `demand_forecasting.csv`, `inventory_monitoring.csv`, and `pricing_optimization.csv`.")
        return None, None, None

# ✅ Load Data
df_demand, df_inventory, df_pricing = load_data()

# 🎯 **Multi-Agent AI System**
class DemandForecastingAgent:
    def __init__(self, df):
        self.df = df

    def train_model(self):
        if "Sales Quantity" in self.df.columns:
            X = self.df.drop(columns=["Sales Quantity"])
            y = self.df["Sales Quantity"]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X.select_dtypes(include=[np.number]), y)
            return model
        return None

class InventoryAgent:
    def __init__(self, df):
        self.df = df

    def check_stock_levels(self):
        if self.df is not None:
            low_stock = self.df[self.df["Stock Levels"] < self.df["Reorder Point"]]
            return low_stock
        return None

class PricingAgent:
    def __init__(self, df):
        self.df = df

    def optimize_prices(self):
        if "Elasticity Index" in self.df.columns:
            self.df["Optimized Price"] = self.df["Price"] * (1 - (self.df["Elasticity Index"] / 10))
            return self.df[["Product ID", "Store ID", "Price", "Optimized Price"]]
        return None

# ✅ Initialize Agents
if df_demand is not None:
    demand_agent = DemandForecastingAgent(df_demand)
    demand_model = demand_agent.train_model()

if df_inventory is not None:
    inventory_agent = InventoryAgent(df_inventory)

if df_pricing is not None:
    pricing_agent = PricingAgent(df_pricing)

# ✅ Streamlit UI
st.title("🤖 AI-Powered Demand, Inventory & Pricing Optimization (Multi-Agent System)")

# 📈 **Demand Forecasting**
st.subheader("📊 Demand Forecasting")
if df_demand is not None:
    st.write(df_demand.head())
    st.success("✅ Demand Forecasting Model Ready!")

# 📦 **Inventory Monitoring**
st.subheader("📦 Inventory Monitoring")
if df_inventory is not None:
    st.write(df_inventory.head())
    low_stock_items = inventory_agent.check_stock_levels()
    if low_stock_items is not None and not low_stock_items.empty:
        st.warning("⚠️ Reorder Needed for These Products!")
        st.write(low_stock_items)

# 💰 **Pricing Optimization**
st.subheader("💰 Pricing Optimization")
if df_pricing is not None:
    optimized_prices = pricing_agent.optimize_prices()
    if optimized_prices is not None:
        st.write(optimized_prices)
        st.success("✅ Optimized Pricing Updated!")

# 📊 **Interactive Data Visualization**
st.subheader("📊 Sales Data Visualization")
if df_demand is not None:
    fig = px.line(df_demand, x="Date", y="Sales Quantity", title="Sales Trends Over Time")
    st.plotly_chart(fig)

st.subheader("📊 Inventory Levels")
if df_inventory is not None:
    fig2 = sns.barplot(x=df_inventory["Product ID"], y=df_inventory["Stock Levels"])
    plt.xticks(rotation=90)
    st.pyplot(fig2.figure)

# 🤖 **AI Chatbot for Business Insights**
st.sidebar.subheader("💬 Ask AI Insights")
user_query = st.sidebar.text_input("Ask a business question")
if user_query:
    client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Replace with your API Key
    response = client.Completion.create(
        model="gpt-4",
        prompt=f"Analyze this business data and answer: {user_query}",
        max_tokens=100
    )
    st.sidebar.write("🧠 AI Response:", response.choices[0].text)

# 📂 **Upload New Data**
st.sidebar.header("🔄 Upload New Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    st.write("📂 New Uploaded Data Preview:")
    st.write(new_df.head())
