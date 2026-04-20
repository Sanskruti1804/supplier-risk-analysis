import pandas as pd

df = pd.read_csv("procurement_data_cleaned.csv")

print(df.head())
print("\nColumns:")
print(df.columns)
print("\nRisk counts:")
print(df["delivery_risk"].value_counts())