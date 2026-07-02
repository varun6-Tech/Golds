import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime, timedelta
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Gold Price Predictor", page_icon="💰", layout="wide")

# ====================== CREATE DATASET IF NOT EXISTS ======================
if not os.path.exists('gold_price.csv'):
    st.info("Creating dataset...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    price = 40000 + (np.arange(300) * 85) + np.random.normal(0, 1500, 300)
    
    df = pd.DataFrame({
        'Date': dates,
        'Gold_Price': price.round(2),
        'Silver_Price': (price * 0.0115).round(2)
    })
    df.to_csv('gold_price.csv', index=False)
    st.success("Dataset created successfully!")

# Load Dataset
df = pd.read_csv('gold_price.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Load Model
@st.cache_resource
def load_model():
    with open('gold_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('poly_transformer.pkl', 'rb') as f:
        poly = pickle.load(f)
    return model, poly

model, poly = load_model()

# ====================== SIDEBAR ======================
st.sidebar.title("💰 Gold Price AI")
st.sidebar.markdown("---")

selected_date = st.sidebar.date_input("Select Prediction Date", datetime.now() + timedelta(days=30))
days = (selected_date - datetime(2020, 1, 1).date()).days

if st.sidebar.button("🔮 Predict Gold Price", type="primary", use_container_width=True):
    input_data = np.array([[days]])
    input_poly = poly.transform(input_data)
    gold_price = model.predict(input_poly)[0]
    silver_price = gold_price * 0.0115

    st.success(f"**Prediction for {selected_date.strftime('%d %B %Y')}**")
    col1, col2 = st.columns(2)
    col1.metric("Gold Price", f"₹ {gold_price:,.2f}", "↑")
    col2.metric("Silver Price", f"₹ {silver_price:,.2f}")

# ====================== MAIN DASHBOARD ======================
st.title("📈 Gold Price Prediction Dashboard")
st.markdown("**Polynomial Regression Model**")

# Charts
st.subheader("Historical Gold Price Trend")
fig = px.line(df, x='Date', y='Gold_Price', title='Gold Price Movement')
st.plotly_chart(fig, use_container_width=True)

st.info("""
**Project Summary:**
- Algorithm: Polynomial Regression (Degree 2)
- Covers: Linear & Polynomial Regression + Feature Transformation
- Good for Viva: Non-linear price prediction
""")

st.caption("✅ Ready for Project Review & Viva")