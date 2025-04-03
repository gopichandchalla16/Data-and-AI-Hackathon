import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Title
st.title("🛒 Multi-Agent AI for Retail Inventory Optimization")

# File Upload & Type Selection
st.sidebar.header("Upload CSV File")
file_type = st.sidebar.selectbox("Select Data Type", ["Demand Forecasting", "Inventory Monitoring"])

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Define required columns based on file type
    if file_type == "Demand Forecasting":
        required_columns = ['Price', 'Promotions', 'Seasonality Factors', 'Sales Quantity']
    else:  # Inventory Monitoring
        required_columns = ['Product ID', 'Stock Level', 'Reorder Point', 'Supplier']

    # Fill missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            st.warning(f"⚠️ Column {col} not found! Filling with default values.")
            df[col] = 0

    st.success("✅ Data Uploaded Successfully!")
    st.write(df.head())  # Show preview of data

    # Demand Forecasting Model
    if file_type == "Demand Forecasting":
        st.subheader("📈 Demand Forecasting")

        # Preprocess Data
        df.fillna(0, inplace=True)  # Handle missing values
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

        # Prepare Data for Model
        X = df.drop(columns=['Sales Quantity'])
        y = df['Sales Quantity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate Model
        mae = mean_absolute_error(y_test, predictions)
        st.write(f"🔍 Mean Absolute Error: {mae:.2f}")

        # Demand Prediction Chart
        fig_demand = px.line(x=y_test.index, y=y_test, labels={'x': 'Index', 'y': 'Sales Quantity'}, title="Actual vs. Predicted Sales")
        fig_demand.add_scatter(x=y_test.index, y=predictions, mode='lines', name='Predicted')
        st.plotly_chart(fig_demand)

    # Inventory Monitoring System
    elif file_type == "Inventory Monitoring":
        st.subheader("📊 Inventory Monitoring")

        # Define Stock Status
        df['Stock Status'] = df.apply(lambda row: 'Low' if row['Stock Level'] < row['Reorder Point'] else 'Sufficient', axis=1)

        # Inventory Distribution Chart
        fig_inventory = px.pie(df, names='Stock Status', title="Stock Distribution")
        st.plotly_chart(fig_inventory)

        # Low Stock Alerts
        low_stock_items = df[df['Stock Status'] == 'Low']
        if not low_stock_items.empty:
            st.warning("⚠️ Low Stock Alert! Reorder Required for These Items:")
            st.dataframe(low_stock_items[['Product ID', 'Stock Level', 'Reorder Point']])
        else:
            st.success("✅ All inventory levels are sufficient!")
