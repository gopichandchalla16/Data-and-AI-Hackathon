import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import difflib

# **📌 Multi-Agent Framework for Retail Optimization**
class DemandAgent:
    """Predicts product demand based on historical data."""
    def __init__(self, data):
        self.data = data

    def find_best_match(self, expected_col, actual_cols):
        matches = difflib.get_close_matches(expected_col, actual_cols, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def preprocess_data(self):
        expected_cols = ['Price', 'Promotions', 'Seasonality Factors', 'Sales Quantity']
        actual_cols = list(self.data.columns)

        col_mapping = {}
        for expected in expected_cols:
            best_match = self.find_best_match(expected, actual_cols)
            if best_match:
                col_mapping[expected] = best_match
            else:
                st.warning(f"⚠️ Column `{expected}` not found! Filling with default values.")
                self.data[expected] = 0  

        self.data.rename(columns=col_mapping, inplace=True)
        self.data.fillna("Unknown", inplace=True)

        label_encoders = {}
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype(str)
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

        except Exception as e:
            st.error(f"❌ Error in Demand Forecasting: {e}")
            return None, None


class InventoryMonitoringAgent:
    """Tracks inventory levels & prevents stockout/overstocking."""
    def __init__(self, data):
        self.data = data

    def check_inventory(self):
        self.data['Stock Status'] = np.where(self.data['Sales Quantity'] > self.data['Predicted Demand'], 'Overstock', 'Optimal')
        self.data['Stock Status'] = np.where(self.data['Sales Quantity'] < self.data['Predicted Demand'], 'Low Stock', self.data['Stock Status'])
        return self.data


class PricingOptimizationAgent:
    """Dynamically adjusts prices based on demand & inventory."""
    def __init__(self, data):
        self.data = data

    def optimize_pricing(self):
        self.data['New Price'] = self.data['Price']
        self.data.loc[self.data['Stock Status'] == 'Overstock', 'New Price'] *= 0.9  
        self.data.loc[self.data['Stock Status'] == 'Low Stock', 'New Price'] *= 1.1  
        return self.data


class SupplyChainAgent:
    """Auto-orders inventory from suppliers if stock is low."""
    def __init__(self, data):
        self.data = data

    def manage_supply_chain(self):
        self.data['Reorder Quantity'] = np.where(self.data['Stock Status'] == 'Low Stock', self.data['Predicted Demand'] - self.data['Sales Quantity'], 0)
        return self.data


# **🚀 Streamlit UI**
st.title("🛒 AI-Powered Retail Inventory Optimization (Multi-Agent System)")
st.markdown("### **Upload Your Retail Data (CSV Format)**")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.write(df.head())

    # **Step 1: Demand Forecasting**
    st.subheader("📈 Demand Forecasting")
    demand_agent = DemandAgent(df)
    df, mae = demand_agent.predict_demand()
    
    if df is not None:
        st.write(f"**Model MAE:** {mae:.2f}")
        fig_demand = px.line(df, x=df.index, y=["Sales Quantity", "Predicted Demand"], title="Actual vs. Predicted Demand")
        st.plotly_chart(fig_demand)

        # **Step 2: Inventory Monitoring**
        st.subheader("📊 Inventory Monitoring")
        inventory_agent = InventoryMonitoringAgent(df)
        df = inventory_agent.check_inventory()
        st.write(df[['Stock Status']].value_counts())

        # **Step 3: Pricing Optimization**
        st.subheader("💰 Dynamic Pricing Optimization")
        pricing_agent = PricingOptimizationAgent(df)
        df = pricing_agent.optimize_pricing()
        st.write(df[['Price', 'New Price']].head())

        # **Step 4: Supply Chain Management**
        st.subheader("🚛 Automated Supply Chain Management")
        supply_chain_agent = SupplyChainAgent(df)
        df = supply_chain_agent.manage_supply_chain()
        st.write(df[['Stock Status', 'Reorder Quantity']].head())

        # **Visualization**
        fig_inventory = px.pie(df, names='Stock Status', title="Stock Distribution"
        st.plotly_chart(fig_inventory)

        fig_pricing = px.scatter(df, x="Price", y="New Price", title="Price Adjustments", color="Stock Status")
        st.plotly_chart(fig_pricing)

        st.success("✅ Multi-Agent AI System Successfully Executed!")
else:
    st.warning("⚠️ Please upload a CSV file.")
