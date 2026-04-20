import pandas as pd

df = pd.read_csv("procurement_data.csv")

df["po_date"] = pd.to_datetime(df["po_date"])
df["promised_date"] = pd.to_datetime(df["promised_date"])
df["actual_date"] = pd.to_datetime(df["actual_date"])

df["delay_days"] = (df["actual_date"] - df["promised_date"]).dt.days
df["delivery_risk"] = (df["delay_days"] > 3).astype(int)
df["fill_rate"] = df["delivered_qty"] / df["ordered_qty"]
df["cost_impact"] = df["po_value"] * df["defect_rate"]

df.to_csv("procurement_data_cleaned.csv", index=False)

print("Cleaned dataset created successfully!")