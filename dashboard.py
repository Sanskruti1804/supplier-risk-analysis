import pandas as pd
import streamlit as st

st.set_page_config(page_title="Supplier Risk Dashboard", layout="wide")

# Load data
df = pd.read_csv("data/procurement_data_cleaned.csv")

st.title("Supplier Risk Prediction Dashboard")
st.write("Interactive procurement analytics dashboard for supplier delivery risk monitoring.")

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

region_options = ["All"] + sorted(df["region"].dropna().unique().tolist())
supplier_options = ["All"] + sorted(df["supplier_name"].dropna().unique().tolist())
category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())

selected_region = st.sidebar.selectbox("Select Region", region_options)
selected_supplier = st.sidebar.selectbox("Select Supplier", supplier_options)
selected_category = st.sidebar.selectbox("Select Category", category_options)
show_only_risky = st.sidebar.checkbox("Show only risky orders")

# -------------------------
# Apply filters
# -------------------------
filtered_df = df.copy()

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_supplier != "All":
    filtered_df = filtered_df[filtered_df["supplier_name"] == selected_supplier]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]

if show_only_risky:
    filtered_df = filtered_df[filtered_df["delivery_risk"] == 1]

# -------------------------
# KPIs
# -------------------------
total_orders = len(filtered_df)
total_suppliers = filtered_df["supplier_name"].nunique()
risky_orders = int(filtered_df["delivery_risk"].sum()) if total_orders > 0 else 0
risk_rate = round((risky_orders / total_orders) * 100, 2) if total_orders > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", total_orders)
col2.metric("Total Suppliers", total_suppliers)
col3.metric("Risky Orders", risky_orders)
col4.metric("Risk Rate (%)", risk_rate)

st.markdown("---")

# -------------------------
# Top risky suppliers
# -------------------------
st.subheader("Top Risky Suppliers")

if total_orders > 0:
    supplier_risk = (
        filtered_df.groupby("supplier_name")["delivery_risk"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    st.dataframe(supplier_risk.head(10), width="stretch")
else:
    st.warning("No data available for selected filters.")

st.markdown("---")

# -------------------------
# Risk by region
# -------------------------
st.subheader("Risk by Region")

if total_orders > 0:
    region_risk = (
        filtered_df.groupby("region")["delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    region_risk["delivery_risk"] = (region_risk["delivery_risk"] * 100).round(2)
    st.bar_chart(region_risk.set_index("region"))
else:
    st.warning("No chart available for selected filters.")

st.markdown("---")

# -------------------------
# Risk by category
# -------------------------
st.subheader("Risk by Category")

if total_orders > 0:
    category_risk = (
        filtered_df.groupby("category")["delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    category_risk["delivery_risk"] = (category_risk["delivery_risk"] * 100).round(2)
    st.bar_chart(category_risk.set_index("category"))
else:
    st.warning("No category data available.")

st.markdown("---")

# -------------------------
# Detailed filtered data
# -------------------------
st.subheader("Filtered Order Data")
if total_orders > 0:
    st.dataframe(filtered_df, width="stretch")
else:
    st.warning("No records match the selected filters.")

st.markdown("---")

# -------------------------
# Images
# -------------------------
st.subheader("Feature Importance")
st.image("outputs/feature_importance.png", width="stretch")

st.markdown("---")

st.subheader("SHAP Explainability")
st.image("outputs/shap_summary.png", width="stretch")


